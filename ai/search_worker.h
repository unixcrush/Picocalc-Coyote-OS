#ifndef COYOTE_AI_SEARCH_WORKER_H
#define COYOTE_AI_SEARCH_WORKER_H

#include <stddef.h>
#include <stdint.h>

#include "ai/search_index.h"

/*
 * Scores int8 vectors against a query on RP2350 core 1, so core 0 can keep the
 * UI and the main loop running.  docs/design/on-device-search.md has the whole
 * ownership and concurrency model; the rules in short:
 *
 *  - Core 0 owns the card.  The worker never opens, reads or closes a file: a
 *    job carries a copy of one block of vectors that core 0 already read, and
 *    the worker only runs search_index_score_block over it.
 *  - Nothing allocates.  The query and the block are copied into the buffers
 *    below at submit time, so the caller's memory is free to reuse at once.
 *  - One job at a time.  IDLE -> QUEUED -> RUNNING -> DONE -> IDLE; a submit
 *    while the state is not IDLE is refused with SEARCH_WORKER_BUSY.
 *  - Core 0 never waits.  Submit, poll and take return at once.
 *
 * This file and search_worker.c have no hardware in them.  Starting core 1 and
 * waking it is search_worker_pico.c, built into the firmware only; the host
 * tests run the worker's side of the protocol by calling
 * search_worker_run_pending() themselves.
 */

#define SEARCH_WORKER_MAX_DIM 1024u
#define SEARCH_WORKER_BLOCK_BYTES SEARCH_IO_CHUNK

typedef enum {
    SEARCH_WORKER_STOPPED = 0, /* not initialized, or shut down */
    SEARCH_WORKER_IDLE,
    SEARCH_WORKER_QUEUED,  /* a job is handed over, core 1 has not started it */
    SEARCH_WORKER_RUNNING, /* core 1 is scoring */
    SEARCH_WORKER_DONE,    /* a result waits to be taken */
} search_worker_state_t;

typedef enum {
    SEARCH_WORKER_OK = 0,
    SEARCH_WORKER_BUSY,          /* a job or an untaken result is in the way */
    SEARCH_WORKER_ERR_ARGUMENT,  /* null pointer, empty or ragged block */
    SEARCH_WORKER_ERR_DIM,       /* dim is 0 or above SEARCH_WORKER_MAX_DIM */
    SEARCH_WORKER_ERR_TOO_LARGE, /* block above SEARCH_WORKER_BLOCK_BYTES */
    SEARCH_WORKER_ERR_NOT_RUNNING,
    SEARCH_WORKER_ERR_NO_RESULT, /* nothing to take */
    SEARCH_WORKER_ERR_START,     /* core 1 could not be started */
} search_worker_status_t;

typedef struct {
    search_status_t status; /* of the scoring, SEARCH_OK when it ran */
    uint32_t first_chunk_id;
    size_t hit_count;
    search_hit_t hits[SEARCH_MAX_HITS];
} search_worker_result_t;

/* Starts core 1 (firmware) or just arms the state machine (host).  Calling it
 * again while running is a no-op returning OK. */
search_worker_status_t search_worker_init(void);

/*
 * Hands one block to the worker: block_length / dim whole vectors, the first
 * being chunk first_chunk_id.  Copies query and block.  Never blocks.
 */
search_worker_status_t search_worker_submit_vector(const int8_t *query, size_t dim, const uint8_t *block,
                                                   size_t block_length, uint32_t first_chunk_id);

/* The state, read with acquire ordering; a result is there when it is DONE. */
search_worker_state_t search_worker_poll(void);

/* Moves a DONE result out and returns the worker to IDLE.  ERR_NO_RESULT in any
 * other state. */
search_worker_status_t search_worker_take_result(search_worker_result_t *result);

/*
 * Stops core 1 so it can be started again by search_worker_init().  Only from
 * IDLE: while a job is in flight or a result is untaken it returns BUSY and
 * changes nothing.  Stopped already is OK.
 */
search_worker_status_t search_worker_shutdown(void);

/* ---- the worker's side ---------------------------------------------------- */

/*
 * What core 1 runs when woken: if a job is QUEUED, scores it and publishes the
 * result.  Returns 1 if it did, 0 if there was nothing queued.  Called by
 * search_worker_pico.c on core 1; the host tests call it in place of core 1.
 */
int search_worker_run_pending(void);

/* ---- platform glue, provided by search_worker_pico.c or a test fake ------- */

/* Launches core 1 so that it calls search_worker_run_pending() each time
 * search_worker_platform_wake() is called.  Returns 0 on success. */
int search_worker_platform_start(void);
/* Wakes core 1; never blocks. */
void search_worker_platform_wake(void);
/* Stops core 1; it is idle when this is called. */
void search_worker_platform_stop(void);

#endif
