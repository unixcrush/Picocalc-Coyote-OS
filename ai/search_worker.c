#include "ai/search_worker.h"

#include <string.h>

/*
 * No <stdio.h>, no malloc, no hardware: see search_worker.h.  The two cores
 * share one job slot.  `state` is the only word they race on, and it is only
 * ever changed by the core that owns the current step:
 *
 *   state        owner        may touch the slot's other fields
 *   STOPPED      core 0       -
 *   IDLE         core 0       yes (submit fills it)
 *   QUEUED       core 1       job (read only)
 *   RUNNING      core 1       job (read), result (write)
 *   DONE         core 0       result (take reads it)
 *
 * A release store hands the slot over and an acquire load takes it back, so
 * everything written before the hand-over is visible after it.
 */

static struct {
    search_worker_state_t state;
    uint16_t dim;
    uint32_t first_chunk_id;
    size_t block_length;
    int8_t query[SEARCH_WORKER_MAX_DIM];
    uint8_t block[SEARCH_WORKER_BLOCK_BYTES];
    search_worker_result_t result;
} slot;

static search_worker_state_t load_state(void) {
    return __atomic_load_n(&slot.state, __ATOMIC_ACQUIRE);
}

static void store_state(search_worker_state_t state) {
    __atomic_store_n(&slot.state, state, __ATOMIC_RELEASE);
}

search_worker_status_t search_worker_init(void) {
    if (load_state() != SEARCH_WORKER_STOPPED) {
        return SEARCH_WORKER_OK;
    }
    if (search_worker_platform_start() != 0) {
        return SEARCH_WORKER_ERR_START;
    }
    store_state(SEARCH_WORKER_IDLE);
    return SEARCH_WORKER_OK;
}

search_worker_status_t search_worker_submit_vector(const int8_t *query, size_t dim, const uint8_t *block,
                                                   size_t block_length, uint32_t first_chunk_id) {
    search_worker_state_t state = load_state();

    if (state == SEARCH_WORKER_STOPPED) {
        return SEARCH_WORKER_ERR_NOT_RUNNING;
    }
    if (state != SEARCH_WORKER_IDLE) {
        return SEARCH_WORKER_BUSY;
    }
    if (query == NULL || block == NULL) {
        return SEARCH_WORKER_ERR_ARGUMENT;
    }
    if (dim == 0u || dim > SEARCH_WORKER_MAX_DIM) {
        return SEARCH_WORKER_ERR_DIM;
    }
    if (block_length > SEARCH_WORKER_BLOCK_BYTES) {
        return SEARCH_WORKER_ERR_TOO_LARGE;
    }
    if (block_length == 0u || block_length % dim != 0u) {
        return SEARCH_WORKER_ERR_ARGUMENT;
    }

    slot.dim = (uint16_t)dim;
    slot.first_chunk_id = first_chunk_id;
    slot.block_length = block_length;
    memcpy(slot.query, query, dim);
    memcpy(slot.block, block, block_length);
    store_state(SEARCH_WORKER_QUEUED);
    search_worker_platform_wake();
    return SEARCH_WORKER_OK;
}

search_worker_state_t search_worker_poll(void) {
    return load_state();
}

search_worker_status_t search_worker_take_result(search_worker_result_t *result) {
    if (result == NULL) {
        return SEARCH_WORKER_ERR_ARGUMENT;
    }
    if (load_state() != SEARCH_WORKER_DONE) {
        return SEARCH_WORKER_ERR_NO_RESULT;
    }
    *result = slot.result;
    store_state(SEARCH_WORKER_IDLE);
    return SEARCH_WORKER_OK;
}

search_worker_status_t search_worker_shutdown(void) {
    search_worker_state_t state = load_state();

    if (state == SEARCH_WORKER_STOPPED) {
        return SEARCH_WORKER_OK;
    }
    if (state != SEARCH_WORKER_IDLE) {
        return SEARCH_WORKER_BUSY;
    }
    /* Core 1 is idle (IDLE is only reached after it has published its result),
     * so it holds nothing that stopping it could leave half done. */
    search_worker_platform_stop();
    store_state(SEARCH_WORKER_STOPPED);
    return SEARCH_WORKER_OK;
}

int search_worker_run_pending(void) {
    search_worker_state_t expected = SEARCH_WORKER_QUEUED;

    /* A wake with no job queued (a late or doubled doorbell) does nothing. */
    if (!__atomic_compare_exchange_n(&slot.state, &expected, SEARCH_WORKER_RUNNING, 0, __ATOMIC_ACQUIRE,
                                     __ATOMIC_RELAXED)) {
        return 0;
    }
    slot.result.first_chunk_id = slot.first_chunk_id;
    slot.result.hit_count = 0u;
    slot.result.status = search_index_score_block(slot.query, slot.dim, slot.block, slot.block_length,
                                                  slot.first_chunk_id, slot.result.hits, SEARCH_MAX_HITS,
                                                  &slot.result.hit_count);
    store_state(SEARCH_WORKER_DONE);
    return 1;
}
