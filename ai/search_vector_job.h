#ifndef COYOTE_AI_SEARCH_VECTOR_JOB_H
#define COYOTE_AI_SEARCH_VECTOR_JOB_H

#include <stddef.h>
#include <stdint.h>

#include "ai/search_index.h"
#include "ai/search_worker.h"

/*
 * Core 0's side of an offloaded vector or hybrid search.  The index is opened,
 * read and closed on core 0 only.  Each search_vector_job_poll() call does at
 * most one bounded card read (whole vectors, up to SEARCH_WORKER_BLOCK_BYTES),
 * hands that block to the core 1 worker and collects the previous block's
 * result, so one call is short and the main loop stays responsive.  The file
 * is never shared with core 1, and the job holds no more than the index's own
 * io buffer.
 *
 * A hybrid job runs its BM25 half inside _begin_hybrid on core 0: that half is
 * card-bound, not CPU-bound, and stays synchronous in this slice.  Only the
 * vector scoring moves to core 1.
 *
 * The caller owns the index and must keep it open, and the card mounted,
 * until the job is DONE, FAILED or cancelled and drained (see _cancel).
 * Not reentrant: one job per index, and the worker serves one job at a time.
 */

typedef enum {
    SEARCH_JOB_IDLE = 0, /* never begun, or begun and refused */
    SEARCH_JOB_PENDING,  /* more polling needed */
    SEARCH_JOB_DONE,     /* search_vector_job_result() has the hits */
    SEARCH_JOB_FAILED,   /* search_vector_job_status() says why */
    SEARCH_JOB_CANCELLED, /* cancelled, and the worker has given its block back */
} search_job_state_t;

typedef struct {
    search_job_state_t state;
    search_status_t status;
    search_index_t *index;
    int8_t query[SEARCH_WORKER_MAX_DIM];
    size_t dim;
    size_t max_hits;
    size_t depth;          /* hits kept from the vector pass */
    uint32_t next_chunk;   /* the next chunk to read */
    int in_flight;         /* a block is with the worker */
    int cancelled;
    int hybrid;
    search_hit_t keyword[SEARCH_MAX_HITS];
    size_t keyword_count;
    search_hit_t hits[SEARCH_MAX_HITS];
    size_t hit_count;
} search_vector_job_t; /* fixed-size, no heap */

/*
 * Starts a vector search of index for query.  Refusals mirror
 * search_index_query_vectors (ERR_NO_VECTORS, ERR_DIM, ERR_FORMAT for a
 * non-int8 index) plus SEARCH_ERR_TOO_LARGE when dim is above
 * SEARCH_WORKER_MAX_DIM; a refusal leaves the job IDLE.
 */
search_status_t search_vector_job_begin(search_vector_job_t *job, search_index_t *index,
                                        const int8_t *query, size_t dim, size_t max_hits);
/*
 * Starts a hybrid search.  Same results and errors as search_index_query_hybrid;
 * on an index without vectors the job is DONE at once with the BM25 hits.
 */
search_status_t search_vector_job_begin_hybrid(search_vector_job_t *job, search_index_t *index,
                                               const char *text, const int8_t *query, size_t dim,
                                               size_t max_hits);
/* One short step.  Never waits for core 1. */
search_job_state_t search_vector_job_poll(search_vector_job_t *job);
/* Why a FAILED job failed (SEARCH_OK otherwise). */
search_status_t search_vector_job_status(const search_vector_job_t *job);
/* The hits of a DONE job.  SEARCH_ERR_ARGUMENT in any other state. */
search_status_t search_vector_job_result(const search_vector_job_t *job, search_hit_t *hits,
                                         size_t max_hits, size_t *hit_count);
/*
 * Gives the search up.  A block already with the worker still has to come back,
 * so keep polling until the state is no longer PENDING; only then may the index
 * be closed or the card handed to USB.
 */
void search_vector_job_cancel(search_vector_job_t *job);

#endif
