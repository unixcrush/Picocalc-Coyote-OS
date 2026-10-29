#include "ai/search_vector_job.h"

#include <string.h>

static void fail(search_vector_job_t *job, search_status_t status) {
    job->state = SEARCH_JOB_FAILED;
    job->status = status;
}

static search_status_t start(search_vector_job_t *job, search_index_t *index, const int8_t *query,
                             size_t dim, size_t max_hits, int hybrid) {
    if (job == NULL || index == NULL || query == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (index->dim == 0u) {
        return SEARCH_ERR_NO_VECTORS;
    }
    if (dim != index->dim) {
        return SEARCH_ERR_DIM;
    }
    if (dim > SEARCH_WORKER_MAX_DIM) {
        return SEARCH_ERR_TOO_LARGE;
    }
    if (index->dtype != SEARCH_DTYPE_INT8 ||
        index->sections[SEARCH_SECTION_VECTORS].len != (uint64_t)index->chunk_count * dim) {
        return SEARCH_ERR_FORMAT;
    }
    /* The worker must have been started: this is a programming error, not a
     * state to wait out. */
    if (search_worker_poll() == SEARCH_WORKER_STOPPED) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (max_hits > SEARCH_MAX_HITS) {
        max_hits = SEARCH_MAX_HITS;
    }
    job->index = index;
    memcpy(job->query, query, dim);
    job->dim = dim;
    job->max_hits = max_hits;
    /* A hybrid fusion reads the vector list SEARCH_MAX_HITS deep. */
    job->depth = hybrid ? SEARCH_MAX_HITS : max_hits;
    job->next_chunk = 0u;
    job->in_flight = 0;
    job->cancelled = 0;
    job->hybrid = hybrid;
    job->hit_count = 0u;
    job->status = SEARCH_OK;
    job->state = SEARCH_JOB_PENDING;
    return SEARCH_OK;
}

search_status_t search_vector_job_begin(search_vector_job_t *job, search_index_t *index,
                                        const int8_t *query, size_t dim, size_t max_hits) {
    search_status_t status;

    if (job != NULL) {
        job->state = SEARCH_JOB_IDLE;
        job->status = SEARCH_OK;
    }
    status = start(job, index, query, dim, max_hits, 0);
    if (status != SEARCH_OK && job != NULL) {
        job->state = SEARCH_JOB_IDLE;
    }
    return status;
}

search_status_t search_vector_job_begin_hybrid(search_vector_job_t *job, search_index_t *index,
                                               const char *text, const int8_t *query, size_t dim,
                                               size_t max_hits) {
    search_status_t status;

    if (job != NULL) {
        job->state = SEARCH_JOB_IDLE;
        job->status = SEARCH_OK;
    }
    if (job == NULL || index == NULL || text == NULL || index->file == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    if (index->dim == 0u) {
        /* As search_index_query_hybrid: BM25 alone, the vector is not looked at. */
        status = search_index_query(index, text, job->hits, max_hits, &job->hit_count);
        if (status == SEARCH_OK) {
            job->state = SEARCH_JOB_DONE;
        }
        return status;
    }
    if (query == NULL) {
        return SEARCH_ERR_ARGUMENT;
    }
    status = start(job, index, query, dim, max_hits, 1);
    if (status != SEARCH_OK) {
        job->state = SEARCH_JOB_IDLE;
        return status;
    }
    /* BM25 is card-bound and runs here, on core 0, before any block is sent. */
    status = search_index_query(index, text, job->keyword, SEARCH_MAX_HITS, &job->keyword_count);
    if (status != SEARCH_OK) {
        job->state = SEARCH_JOB_IDLE;
        return status;
    }
    return SEARCH_OK;
}

static void finish(search_vector_job_t *job) {
    if (job->hybrid) {
        search_hit_t fused[SEARCH_MAX_HITS];
        size_t fused_count = 0u;

        search_index_fuse(job->keyword, job->keyword_count, job->hits, job->hit_count, fused,
                          job->max_hits, &fused_count);
        memcpy(job->hits, fused, fused_count * sizeof(fused[0]));
        job->hit_count = fused_count;
    }
    job->state = SEARCH_JOB_DONE;
}

search_job_state_t search_vector_job_poll(search_vector_job_t *job) {
    search_worker_result_t result;
    size_t chunks = 0u;
    size_t i;
    search_status_t status;

    if (job == NULL || job->state != SEARCH_JOB_PENDING) {
        return job == NULL ? SEARCH_JOB_IDLE : job->state;
    }

    if (job->in_flight) {
        if (search_worker_poll() != SEARCH_WORKER_DONE) {
            return SEARCH_JOB_PENDING;
        }
        if (search_worker_take_result(&result) != SEARCH_WORKER_OK) {
            fail(job, SEARCH_ERR_ARGUMENT);
            return job->state;
        }
        job->in_flight = 0;
        if (job->cancelled) {
            job->state = SEARCH_JOB_CANCELLED;
            return job->state;
        }
        if (result.status != SEARCH_OK) {
            fail(job, result.status);
            return job->state;
        }
        /* One block in flight at a time, so blocks arrive in rising chunk order
         * and equal scores keep the lower chunk id first. */
        for (i = 0u; i < result.hit_count; i++) {
            search_index_keep_hit(job->hits, job->depth, &job->hit_count, result.hits[i].chunk_id,
                                  result.hits[i].score_q16);
        }
    }
    if (job->cancelled) {
        job->state = SEARCH_JOB_CANCELLED;
        return job->state;
    }
    if (job->next_chunk >= job->index->chunk_count) {
        finish(job);
        return job->state;
    }
    if (search_worker_poll() != SEARCH_WORKER_IDLE) {
        /* Someone else's job or an untaken result: try again next poll. */
        return SEARCH_JOB_PENDING;
    }

    status = search_index_read_vector_block(job->index, job->next_chunk, job->index->io,
                                            SEARCH_WORKER_BLOCK_BYTES, &chunks);
    if (status != SEARCH_OK) {
        fail(job, status);
        return job->state;
    }
    if (search_worker_submit_vector(job->query, job->dim, job->index->io, chunks * job->dim,
                                    job->next_chunk) != SEARCH_WORKER_OK) {
        fail(job, SEARCH_ERR_ARGUMENT);
        return job->state;
    }
    job->next_chunk += (uint32_t)chunks;
    job->in_flight = 1;
    return SEARCH_JOB_PENDING;
}

search_status_t search_vector_job_status(const search_vector_job_t *job) {
    return job == NULL ? SEARCH_ERR_ARGUMENT : job->status;
}

search_status_t search_vector_job_result(const search_vector_job_t *job, search_hit_t *hits,
                                         size_t max_hits, size_t *hit_count) {
    size_t count;

    if (hit_count != NULL) {
        *hit_count = 0u;
    }
    if (job == NULL || hits == NULL || hit_count == NULL || job->state != SEARCH_JOB_DONE) {
        return SEARCH_ERR_ARGUMENT;
    }
    count = job->hit_count < max_hits ? job->hit_count : max_hits;
    memcpy(hits, job->hits, count * sizeof(hits[0]));
    *hit_count = count;
    return SEARCH_OK;
}

void search_vector_job_cancel(search_vector_job_t *job) {
    if (job == NULL || job->state != SEARCH_JOB_PENDING) {
        return;
    }
    job->cancelled = 1;
    if (!job->in_flight) {
        job->state = SEARCH_JOB_CANCELLED;
    }
}
