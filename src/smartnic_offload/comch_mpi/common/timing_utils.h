#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <stdint.h>
#include <time.h>
#include <sched.h>

static inline uint64_t now_monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void cpu_relax(void)
{
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* ---- Hybrid polling knobs ---- */
#define HYBRID_BUSY_POLL_NS   (200ULL * 1000)  /* 200us: ここまではスピン */
#define HYBRID_SLEEP_NS       (1000)           /* 1us: 最終手段の sleep */
#define HYBRID_YIELD_EVERY    (256)            /* yield を入れる頻度(ループ回数) */

/* progressed==0 のときだけ呼ぶことを想定 */
static inline void hybrid_wait(uint64_t start_ns, uint32_t *spin_cnt)
{
    uint64_t now = now_monotonic_ns();
    uint64_t elapsed = now - start_ns;

    if (elapsed < HYBRID_BUSY_POLL_NS) {
        /* busy-poll 区間: 低レイテンシ優先 */
        cpu_relax();
        return;
    }

    /* 省電力区間: まずは軽い譲歩 */
    (*spin_cnt)++;
    if (((*spin_cnt) & (HYBRID_YIELD_EVERY - 1)) == 0) {
        sched_yield();
        return;
    }

    /* 最終手段: ごく短い sleep */
    struct timespec ts = {.tv_sec = 0, .tv_nsec = HYBRID_SLEEP_NS};
    nanosleep(&ts, NULL);
}

#endif /* TIMING_UTILS_H */
