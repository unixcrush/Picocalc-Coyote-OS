#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ai/search_index.h"
#include "ai/search_vector_job.h"
#include "ai/search_worker.h"
#include "test_util.h"

#define WORKER_FIXTURE_FILE "search_worker_test.idx"

/*
 * The platform seam, faked: "core 1" is the test calling
 * search_worker_run_pending() at the moments it chooses, so every state of the
 * protocol can be looked at.  The real glue (search_worker_pico.c) is
 * firmware-only and not covered here.
 */
static int fake_start_calls;
static int fake_stop_calls;
static int fake_wake_calls;
static int fake_start_fails;

int search_worker_platform_start(void) {
    fake_start_calls++;
    return fake_start_fails ? -1 : 0;
}

void search_worker_platform_wake(void) {
    fake_wake_calls++;
}

void search_worker_platform_stop(void) {
    fake_stop_calls++;
}

static void reset_fake(void) {
    fake_start_calls = fake_stop_calls = fake_wake_calls = 0;
    fake_start_fails = 0;
}

/* ---- a small CYX1 index: N vectors of dim, one BM25 term ------------------ */

static void w16(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void w32(uint8_t *p, uint32_t v) {
    w16(p, v);
    w16(p + 2, v >> 16);
}

static uint32_t crc32_bytes(const uint8_t *data, size_t length) {
    uint32_t crc = 0xFFFFFFFFu;
    size_t i;
    int bit;

    for (i = 0u; i < length; i++) {
        crc ^= data[i];
        for (bit = 0; bit < 8; bit++) {
            crc = (crc & 1u) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }
    }
    return ~crc;
}

static void set_sec(uint8_t *file, unsigned section, size_t off, size_t len) {
    w32(file + 96 + 8u * section, (uint32_t)off);
    w32(file + 100 + 8u * section, (uint32_t)len);
}

static uint32_t fnv(const char *text) {
    uint32_t hash = 2166136261u;

    for (; *text != '\0'; text++) {
        hash = (hash ^ (uint8_t)*text) * 16777619u;
    }
    return hash;
}

/* Chunk c's element i.  Chunks 4 and 5 are equal, so a tie falls on a block
 * boundary for the dims below. */
static int8_t element(size_t chunk, size_t i) {
    size_t c = chunk == 5u ? 4u : chunk;

    return (int8_t)((int)((c * 37u + i * 11u) % 255u) - 127);
}

static uint8_t file_bytes[64u * 1024u];

/* "radio" is in chunk 3 (tf 2) and chunk 7 (tf 1). */
static void build_index(size_t chunks, size_t dim) {
    size_t terms_off = SEARCH_INDEX_HEADER_SIZE;
    size_t postings_off = terms_off + 12u;
    size_t docs_off = postings_off + 12u;
    size_t vectors_off = docs_off + chunks * 12u;
    size_t end = vectors_off + chunks * dim;
    size_t c;
    size_t i;

    memset(file_bytes, 0, sizeof(file_bytes));
    memcpy(file_bytes, "CYX1", 4u);
    w16(file_bytes + 4, 1u);
    w16(file_bytes + 6, SEARCH_FLAG_NORMALIZED | SEARCH_FLAG_INT8);
    w16(file_bytes + 8, (uint32_t)dim);
    w16(file_bytes + 10, dim == 0u ? SEARCH_DTYPE_NONE : SEARCH_DTYPE_INT8);
    w32(file_bytes + 12, (uint32_t)chunks);
    w32(file_bytes + 16, 1u);
    w32(file_bytes + 20, 256u);
    w32(file_bytes + terms_off, fnv("radio"));
    w32(file_bytes + terms_off + 4, 0u);
    w16(file_bytes + terms_off + 8, 2u);
    w32(file_bytes + postings_off, 3u);
    w16(file_bytes + postings_off + 4, 2u);
    w32(file_bytes + postings_off + 6, 7u);
    w16(file_bytes + postings_off + 10, 1u);
    for (c = 0u; c < chunks; c++) {
        w16(file_bytes + docs_off + c * 12u + 8u, 1u);
    }
    for (c = 0u; c < chunks; c++) {
        for (i = 0u; i < dim; i++) {
            file_bytes[vectors_off + c * dim + i] = (uint8_t)element(c, i);
        }
    }
    set_sec(file_bytes, SEARCH_SECTION_TERMS, terms_off, 12u);
    set_sec(file_bytes, SEARCH_SECTION_POSTINGS, postings_off, 12u);
    set_sec(file_bytes, SEARCH_SECTION_DOCS, docs_off, chunks * 12u);
    set_sec(file_bytes, SEARCH_SECTION_TEXT, vectors_off, 0u);
    set_sec(file_bytes, SEARCH_SECTION_VECTORS, vectors_off, chunks * dim);
    set_sec(file_bytes, SEARCH_SECTION_RESERVED, end, 0u);
    w32(file_bytes + 24, crc32_bytes(file_bytes + 28, end - 28u));
    {
        FILE *file = fopen(WORKER_FIXTURE_FILE, "wb");

        if (file != NULL) {
            fwrite(file_bytes, 1u, end, file);
            fclose(file);
        }
    }
}

static search_index_t worker_index;
static int8_t query_vec[SEARCH_WORKER_MAX_DIM + 8u];

static void make_query(size_t dim) {
    size_t i;

    for (i = 0u; i < dim; i++) {
        query_vec[i] = (int8_t)((int)((i * 5u + 3u) % 255u) - 127);
    }
}

/* ---- protocol ------------------------------------------------------------- */

static void check_stopped_and_init(void) {
    static const uint8_t block[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    static const int8_t query[4] = {1, 2, 3, 4};
    search_worker_result_t result;

    reset_fake();
    /* Before init nothing is accepted and nothing is there to take. */
    CHECK(search_worker_poll() == SEARCH_WORKER_STOPPED);
    CHECK(search_worker_submit_vector(query, 4u, block, 8u, 0u) == SEARCH_WORKER_ERR_NOT_RUNNING);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_ERR_NO_RESULT);
    CHECK(search_worker_run_pending() == 0);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK && fake_stop_calls == 0);

    /* A core that will not start leaves the worker stopped. */
    fake_start_fails = 1;
    CHECK(search_worker_init() == SEARCH_WORKER_ERR_START);
    CHECK(search_worker_poll() == SEARCH_WORKER_STOPPED);
    fake_start_fails = 0;

    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE);
    CHECK(fake_start_calls == 2);
    /* A second init does not start core 1 again. */
    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    CHECK(fake_start_calls == 2);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_transitions(void) {
    static const int8_t query[4] = {127, 0, 0, 0};
    /* Three vectors of dim 4, the cosine with query rising to chunk 9. */
    static const uint8_t block[12] = {10, 0, 0, 0, 100, 0, 0, 0, 50, 0, 0, 0};
    uint8_t scratch[12];
    search_worker_result_t result;

    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_ERR_NO_RESULT);
    CHECK(search_worker_run_pending() == 0); /* nothing queued: a stray wake */

    /* The caller's block is copied: scribbling on it after submit changes nothing. */
    memcpy(scratch, block, sizeof(scratch));
    CHECK(search_worker_submit_vector(query, 4u, scratch, sizeof(scratch), 7u) == SEARCH_WORKER_OK);
    memset(scratch, 0, sizeof(scratch));
    CHECK(fake_wake_calls == 1);
    CHECK(search_worker_poll() == SEARCH_WORKER_QUEUED);

    /* Rejected while busy, in every busy state, without disturbing the job. */
    CHECK(search_worker_submit_vector(query, 4u, block, sizeof(block), 0u) == SEARCH_WORKER_BUSY);
    CHECK(fake_wake_calls == 1);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_ERR_NO_RESULT);
    CHECK(search_worker_poll() == SEARCH_WORKER_QUEUED);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_BUSY && fake_stop_calls == 0);

    CHECK(search_worker_run_pending() == 1);
    CHECK(search_worker_poll() == SEARCH_WORKER_DONE);
    CHECK(search_worker_run_pending() == 0); /* a doubled wake is harmless */
    CHECK(search_worker_poll() == SEARCH_WORKER_DONE);
    /* An untaken result also blocks a new job and a shutdown. */
    CHECK(search_worker_submit_vector(query, 4u, block, sizeof(block), 0u) == SEARCH_WORKER_BUSY);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_BUSY && fake_stop_calls == 0);

    CHECK(search_worker_take_result(NULL) == SEARCH_WORKER_ERR_ARGUMENT);
    CHECK(search_worker_poll() == SEARCH_WORKER_DONE); /* a refused take keeps it */
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK);
    CHECK(result.status == SEARCH_OK && result.first_chunk_id == 7u && result.hit_count == 3u);
    CHECK(result.hits[0].chunk_id == 8u && result.hits[0].score_q16 == 65536 * 100 / 127);
    CHECK(result.hits[1].chunk_id == 9u && result.hits[2].chunk_id == 7u);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_ERR_NO_RESULT); /* only once */

    /* IDLE again: the next job goes through. */
    CHECK(search_worker_submit_vector(query, 4u, block, 4u, 0u) == SEARCH_WORKER_OK);
    CHECK(search_worker_run_pending() == 1);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK && result.hit_count == 1u);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_bounds(void) {
    static int8_t query[SEARCH_WORKER_MAX_DIM + 1u];
    static uint8_t block[SEARCH_WORKER_BLOCK_BYTES + 4u];

    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);

    CHECK(search_worker_submit_vector(NULL, 4u, block, 8u, 0u) == SEARCH_WORKER_ERR_ARGUMENT);
    CHECK(search_worker_submit_vector(query, 4u, NULL, 8u, 0u) == SEARCH_WORKER_ERR_ARGUMENT);
    CHECK(search_worker_submit_vector(query, 0u, block, 8u, 0u) == SEARCH_WORKER_ERR_DIM);
    CHECK(search_worker_submit_vector(query, SEARCH_WORKER_MAX_DIM + 1u, block, SEARCH_WORKER_MAX_DIM + 1u, 0u) ==
          SEARCH_WORKER_ERR_DIM);
    CHECK(search_worker_submit_vector(query, 4u, block, 0u, 0u) == SEARCH_WORKER_ERR_ARGUMENT);
    CHECK(search_worker_submit_vector(query, 4u, block, 6u, 0u) == SEARCH_WORKER_ERR_ARGUMENT);
    CHECK(search_worker_submit_vector(query, 4u, block, SEARCH_WORKER_BLOCK_BYTES + 4u, 0u) ==
          SEARCH_WORKER_ERR_TOO_LARGE);
    /* Every refusal left the worker idle and woke nobody. */
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE && fake_wake_calls == 0);

    /* The limits themselves are accepted: the widest dim in one vector, and a
     * full block of narrow ones. */
    CHECK(search_worker_submit_vector(query, SEARCH_WORKER_MAX_DIM, block, SEARCH_WORKER_MAX_DIM, 0u) ==
          SEARCH_WORKER_OK);
    CHECK(search_worker_run_pending() == 1);
    {
        search_worker_result_t result;

        CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK && result.hit_count == 1u);
    }
    CHECK(search_worker_submit_vector(query, 4u, block, SEARCH_WORKER_BLOCK_BYTES, 0u) == SEARCH_WORKER_OK);
    CHECK(search_worker_run_pending() == 1);
    {
        search_worker_result_t result;

        CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK);
        CHECK(result.hit_count == SEARCH_MAX_HITS); /* 1024 vectors, the best 8 kept */
    }

    /* Chunk ids that would pass 2^32 - 1 reach the worker and come back as a
     * scoring error rather than wrapping. */
    CHECK(search_worker_submit_vector(query, 4u, block, 8u, UINT32_MAX) == SEARCH_WORKER_OK);
    CHECK(search_worker_run_pending() == 1);
    {
        search_worker_result_t result;

        CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK);
        CHECK(result.status == SEARCH_ERR_ARGUMENT && result.hit_count == 0u);
    }
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_shutdown(void) {
    static const int8_t query[4] = {1, 1, 1, 1};
    static const uint8_t block[4] = {1, 1, 1, 1};
    search_worker_result_t result;

    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
    CHECK(fake_stop_calls == 1 && search_worker_poll() == SEARCH_WORKER_STOPPED);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK && fake_stop_calls == 1);
    CHECK(search_worker_submit_vector(query, 4u, block, 4u, 0u) == SEARCH_WORKER_ERR_NOT_RUNNING);
    CHECK(search_worker_run_pending() == 0);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_ERR_NO_RESULT);

    /* It can be started again, and works. */
    CHECK(search_worker_init() == SEARCH_WORKER_OK && fake_start_calls == 2);
    CHECK(search_worker_submit_vector(query, 4u, block, 4u, 0u) == SEARCH_WORKER_OK);
    CHECK(search_worker_run_pending() == 1);
    CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK && result.hit_count == 1u);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

/* ---- the job driver over a real index -------------------------------------- */

/* Polls the job, running "core 1" only every `lag`-th poll, so the driver
 * is seen to wait on a block without blocking.  Returns the final state. */
static search_job_state_t drive(search_vector_job_t *job, int lag, int *polls) {
    search_job_state_t state;
    int n = 0;

    *polls = 0;
    do {
        state = search_vector_job_poll(job);
        (*polls)++;
        if (++n % lag == 0) {
            search_worker_run_pending();
        }
    } while (state == SEARCH_JOB_PENDING && *polls < 100000);
    return state;
}

static void same_hits(const search_hit_t *a, const search_hit_t *b, size_t count) {
    size_t i;

    for (i = 0u; i < count; i++) {
        CHECK(a[i].chunk_id == b[i].chunk_id && a[i].score_q16 == b[i].score_q16);
    }
}

static void check_job_matches_sync(size_t chunks, size_t dim, int lag) {
    search_hit_t expected[SEARCH_MAX_HITS];
    search_hit_t got[SEARCH_MAX_HITS];
    static search_vector_job_t storage;
    search_vector_job_t *job = &storage;
    size_t expected_count = 0u;
    size_t got_count = 0u;
    int polls;

    build_index(chunks, dim);
    make_query(dim);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);
    CHECK(search_index_query_vectors(&worker_index, (const int8_t *)query_vec, dim, expected,
                                     SEARCH_MAX_HITS, &expected_count) == SEARCH_OK);

    CHECK(search_vector_job_begin(job, &worker_index, query_vec, dim, SEARCH_MAX_HITS) == SEARCH_OK);
    if (chunks > 0u) { /* an empty index is done on the first poll */
        CHECK(search_vector_job_poll(job) == SEARCH_JOB_PENDING);
        CHECK(search_vector_job_result(job, got, SEARCH_MAX_HITS, &got_count) == SEARCH_ERR_ARGUMENT);
    }
    CHECK(drive(job, lag, &polls) == SEARCH_JOB_DONE);
    CHECK(search_vector_job_status(job) == SEARCH_OK);
    CHECK(search_vector_job_result(job, got, SEARCH_MAX_HITS, &got_count) == SEARCH_OK);
    CHECK(got_count == expected_count);
    same_hits(got, expected, expected_count);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE);
    /* With a slow core 1 the job waited across polls instead of blocking. */
    CHECK(lag == 1 || polls > lag);
    search_index_close(&worker_index);
}

static void check_job(void) {
    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);

    check_job_matches_sync(40u, 4u, 1);   /* one block */
    check_job_matches_sync(40u, 4u, 3);
    check_job_matches_sync(13u, 768u, 1); /* five vectors a block: 3 blocks, a tie spans two */
    check_job_matches_sync(13u, 768u, 4);
    check_job_matches_sync(9u, 1024u, 2); /* four a block */
    check_job_matches_sync(1u, 1024u, 1);
    check_job_matches_sync(0u, 4u, 1);    /* no chunks: done with no hits */

    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_job_refusals(void) {
    static search_vector_job_t job;
    search_hit_t hits[SEARCH_MAX_HITS];
    size_t count = 5u;

    reset_fake();
    build_index(10u, 8u);
    make_query(8u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);

    /* Worker not started: a programming error, refused up front. */
    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 8u, 8u) == SEARCH_ERR_ARGUMENT);
    CHECK(search_worker_init() == SEARCH_WORKER_OK);

    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 7u, 8u) == SEARCH_ERR_DIM);
    CHECK(job.state == SEARCH_JOB_IDLE && search_vector_job_poll(&job) == SEARCH_JOB_IDLE);
    CHECK(search_vector_job_begin(&job, &worker_index, NULL, 8u, 8u) == SEARCH_ERR_ARGUMENT);
    CHECK(search_vector_job_begin(NULL, &worker_index, query_vec, 8u, 8u) == SEARCH_ERR_ARGUMENT);
    CHECK(search_vector_job_result(&job, hits, 8u, &count) == SEARCH_ERR_ARGUMENT && count == 0u);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE && fake_wake_calls == 0);

    /* The worker is busy with someone else's job: the driver waits, then goes. */
    {
        static const uint8_t other[8] = {0};
        int polls;

        CHECK(search_worker_submit_vector((const int8_t *)other, 8u, other, 8u, 0u) == SEARCH_WORKER_OK);
        CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 8u, 8u) == SEARCH_OK);
        CHECK(search_vector_job_poll(&job) == SEARCH_JOB_PENDING);
        CHECK(job.next_chunk == 0u && !job.in_flight);
        search_worker_run_pending();
        {
            search_worker_result_t result;

            CHECK(search_worker_take_result(&result) == SEARCH_WORKER_OK);
        }
        CHECK(drive(&job, 1, &polls) == SEARCH_JOB_DONE);
    }

    /* An index without vectors. */
    search_index_close(&worker_index);
    build_index(10u, 0u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);
    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 8u, 8u) == SEARCH_ERR_NO_VECTORS);
    search_index_close(&worker_index);

    /* dim above the worker's block: the card side could read it, the worker cannot. */
    build_index(2u, 1500u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);
    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 1500u, 8u) == SEARCH_ERR_TOO_LARGE);
    search_index_close(&worker_index);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_job_cancel(void) {
    static search_vector_job_t job;
    int polls;

    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    build_index(40u, 768u);
    make_query(768u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);

    /* Cancelled before a block went out: over at once. */
    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 768u, 8u) == SEARCH_OK);
    search_vector_job_cancel(&job);
    CHECK(job.state == SEARCH_JOB_CANCELLED && search_worker_poll() == SEARCH_WORKER_IDLE);

    /* Cancelled with a block out: stays PENDING until the worker gives it back,
     * after which the worker is free and nothing more is read. */
    CHECK(search_vector_job_begin(&job, &worker_index, query_vec, 768u, 8u) == SEARCH_OK);
    CHECK(search_vector_job_poll(&job) == SEARCH_JOB_PENDING);
    search_vector_job_cancel(&job);
    CHECK(job.state == SEARCH_JOB_PENDING);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_BUSY);
    CHECK(search_vector_job_poll(&job) == SEARCH_JOB_PENDING); /* core 1 has not run */
    CHECK(drive(&job, 1, &polls) == SEARCH_JOB_CANCELLED);
    CHECK(search_worker_poll() == SEARCH_WORKER_IDLE && job.next_chunk == 5u);
    search_vector_job_cancel(&job); /* no effect on a finished job */
    CHECK(job.state == SEARCH_JOB_CANCELLED);

    search_index_close(&worker_index);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_job_hybrid(void) {
    static search_vector_job_t job;
    search_hit_t expected[SEARCH_MAX_HITS];
    search_hit_t got[SEARCH_MAX_HITS];
    size_t expected_count = 0u;
    size_t got_count = 0u;
    int polls;
    size_t depth;

    reset_fake();
    CHECK(search_worker_init() == SEARCH_WORKER_OK);
    build_index(30u, 16u);
    make_query(16u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);

    for (depth = 1u; depth <= SEARCH_MAX_HITS; depth += 7u) {
        CHECK(search_index_query_hybrid(&worker_index, "radio", query_vec, 16u, expected, depth,
                                        &expected_count) == SEARCH_OK);
        CHECK(search_vector_job_begin_hybrid(&job, &worker_index, "radio", query_vec, 16u, depth) == SEARCH_OK);
        CHECK(drive(&job, 2, &polls) == SEARCH_JOB_DONE);
        CHECK(search_vector_job_result(&job, got, SEARCH_MAX_HITS, &got_count) == SEARCH_OK);
        CHECK(got_count == expected_count && got_count > 0u);
        same_hits(got, expected, expected_count);
    }
    /* The caller's cap on the result is honoured. */
    CHECK(search_vector_job_result(&job, got, 2u, &got_count) == SEARCH_OK && got_count == 2u);

    /* Argument and dimension errors are those of the synchronous call. */
    CHECK(search_vector_job_begin_hybrid(&job, &worker_index, "radio", query_vec, 15u, 8u) == SEARCH_ERR_DIM);
    CHECK(search_vector_job_begin_hybrid(&job, &worker_index, "radio", NULL, 16u, 8u) == SEARCH_ERR_ARGUMENT);
    CHECK(search_vector_job_begin_hybrid(&job, &worker_index, NULL, query_vec, 16u, 8u) == SEARCH_ERR_ARGUMENT);
    CHECK(job.state == SEARCH_JOB_IDLE);
    search_index_close(&worker_index);

    /* No vectors: the BM25 answer at once, whatever vector it is handed. */
    build_index(30u, 0u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);
    CHECK(search_index_query_hybrid(&worker_index, "radio", NULL, 0u, expected, 8u, &expected_count) == SEARCH_OK);
    CHECK(search_vector_job_begin_hybrid(&job, &worker_index, "radio", NULL, 0u, 8u) == SEARCH_OK);
    CHECK(job.state == SEARCH_JOB_DONE && search_worker_poll() == SEARCH_WORKER_IDLE);
    CHECK(search_vector_job_result(&job, got, 8u, &got_count) == SEARCH_OK);
    CHECK(got_count == expected_count && got_count == 2u);
    same_hits(got, expected, expected_count);
    search_index_close(&worker_index);
    CHECK(search_worker_shutdown() == SEARCH_WORKER_OK);
}

static void check_block_api(void) {
    static const int8_t query[2] = {127, 0};
    static const uint8_t block[6] = {5, 0, 100, 0, 100, 0};
    search_hit_t hits[SEARCH_MAX_HITS];
    size_t count = 9u;
    uint8_t buffer[64];

    /* The ragged and the null are refused; ties keep the lower chunk id. */
    CHECK(search_index_score_block(query, 2u, block, 5u, 0u, hits, 8u, &count) == SEARCH_ERR_ARGUMENT && count == 0u);
    CHECK(search_index_score_block(query, 0u, block, 6u, 0u, hits, 8u, &count) == SEARCH_ERR_ARGUMENT);
    CHECK(search_index_score_block(NULL, 2u, block, 6u, 0u, hits, 8u, &count) == SEARCH_ERR_ARGUMENT);
    CHECK(search_index_score_block(query, 2u, block, 6u, 0u, hits, 8u, NULL) == SEARCH_ERR_ARGUMENT);
    CHECK(search_index_score_block(query, 2u, block, 6u, UINT32_MAX - 1u, hits, 8u, &count) == SEARCH_ERR_ARGUMENT);
    CHECK(search_index_score_block(query, 2u, block, 6u, UINT32_MAX - 2u, hits, 8u, &count) == SEARCH_OK && count == 3u);
    CHECK(search_index_score_block(query, 2u, block, 6u, 10u, hits, 8u, &count) == SEARCH_OK && count == 3u);
    CHECK(hits[0].chunk_id == 11u && hits[1].chunk_id == 12u && hits[2].chunk_id == 10u);
    CHECK(search_index_score_block(query, 2u, block, 6u, 10u, hits, 1u, &count) == SEARCH_OK && count == 1u);

    /* The block reader. */
    build_index(10u, 8u);
    CHECK(search_index_open(&worker_index, WORKER_FIXTURE_FILE) == SEARCH_OK);
    CHECK(search_index_read_vector_block(&worker_index, 0u, buffer, 20u, &count) == SEARCH_OK && count == 2u);
    CHECK((int8_t)buffer[3] == element(0u, 3u) && (int8_t)buffer[8 + 3] == element(1u, 3u));
    CHECK(search_index_read_vector_block(&worker_index, 9u, buffer, 64u, &count) == SEARCH_OK && count == 1u);
    CHECK(search_index_read_vector_block(&worker_index, 10u, buffer, 64u, &count) == SEARCH_OK && count == 0u);
    CHECK(search_index_read_vector_block(&worker_index, 0u, buffer, 7u, &count) == SEARCH_ERR_TOO_LARGE);
    CHECK(search_index_read_vector_block(&worker_index, 0u, NULL, 64u, &count) == SEARCH_ERR_ARGUMENT);
    search_index_close(&worker_index);
    CHECK(search_index_read_vector_block(&worker_index, 0u, buffer, 64u, &count) == SEARCH_ERR_ARGUMENT);
}

void test_search_worker(void);
void test_search_worker(void) {
    check_block_api();
    check_stopped_and_init();
    check_transitions();
    check_bounds();
    check_shutdown();
    check_job();
    check_job_refusals();
    check_job_cancel();
    check_job_hybrid();
    remove(WORKER_FIXTURE_FILE);
}
