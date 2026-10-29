#include "ai/search_worker.h"

#include "pico/multicore.h"

/*
 * Firmware-only glue: core 1 sleeps in the multicore FIFO's blocking pop (a
 * WFE loop inside the SDK), core 0 rings it with one word per job.  Only one
 * job is ever in flight, so the 8-deep FIFO cannot fill.  Core 1 sends nothing
 * back: core 0 polls the state word instead of waiting on an interrupt.
 *
 * Not hardware-tested; see docs/design/on-device-search.md.
 */

#define DOORBELL 0x53574B31u /* 'SWK1' */

/* Never returns: search_worker_platform_stop() resets the core instead. */

static void core1_main(void) {
    /* Lets core 0 pause this core if it ever writes flash; nothing does today. */
    multicore_lockout_victim_init();
    for (;;) {
        uint32_t word = multicore_fifo_pop_blocking();

        if (word == DOORBELL) {
            search_worker_run_pending();
        }
    }
}

int search_worker_platform_start(void) {
    /* Clears any earlier launch, so a re-init after a shutdown starts clean. */
    multicore_reset_core1();
    multicore_fifo_drain();
    multicore_launch_core1(core1_main);
    return 0;
}

void search_worker_platform_wake(void) {
    /* Zero timeout: core 0 never waits on core 1. */
    (void)multicore_fifo_push_timeout_us(DOORBELL, 0u);
}

void search_worker_platform_stop(void) {
    multicore_reset_core1();
    multicore_fifo_drain();
}
