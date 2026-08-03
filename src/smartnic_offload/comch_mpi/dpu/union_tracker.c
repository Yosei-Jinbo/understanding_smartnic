#define _GNU_SOURCE
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "union_tracker.h"

/* ============================================================
 * DPU Union Tracker (interval union で AG/RS/overlap を wall で測る)
 *
 * 目的: AG と RS が並列実行されても double-count せず、
 *       "pure AG wall busy" / "pure RS wall busy" / "overlap" を正しく取る。
 *
 * 方式: active counter 方式 (online interval union)。
 *   - ag_active/rs_active: 現在走っている AG/RS の個数 (並列 inflight)
 *   - ag_busy_start: ag_active が 0→1 になった時刻
 *   - both_start: (ag_active>0 AND rs_active>0) になった時刻
 *   - 各 total_*_ns: 累積 busy 時間 (interval union)
 *
 * Sum invariant: AG + RS − overlap = ANY (AG or RS active wall time)
 * ============================================================ */
static pthread_mutex_t g_union_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_union_ag_active = 0;
static int g_union_rs_active = 0;
static uint64_t g_union_ag_busy_start_ns = 0;
static uint64_t g_union_rs_busy_start_ns = 0;
static uint64_t g_union_both_start_ns = 0;
static uint64_t g_union_ag_busy_total_ns = 0;
static uint64_t g_union_rs_busy_total_ns = 0;
static uint64_t g_union_both_total_ns = 0;
static uint64_t g_union_ag_count = 0;
static uint64_t g_union_rs_count = 0;
static uint64_t g_union_max_ag_inflight = 0;
static uint64_t g_union_max_rs_inflight = 0;
static int g_union_printed = 0;  /* 重複出力防止 */

static inline uint64_t union_now_ns(void) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
}

void dpu_union_tracker_enter(bool is_ag) {
    uint64_t t = union_now_ns();
    pthread_mutex_lock(&g_union_lock);

    bool was_both = (g_union_ag_active > 0) && (g_union_rs_active > 0);

    if (is_ag) {
        if (g_union_ag_active == 0) g_union_ag_busy_start_ns = t;
        g_union_ag_active++;
        g_union_ag_count++;
        if ((uint64_t)g_union_ag_active > g_union_max_ag_inflight)
            g_union_max_ag_inflight = (uint64_t)g_union_ag_active;
    } else {
        if (g_union_rs_active == 0) g_union_rs_busy_start_ns = t;
        g_union_rs_active++;
        g_union_rs_count++;
        if ((uint64_t)g_union_rs_active > g_union_max_rs_inflight)
            g_union_max_rs_inflight = (uint64_t)g_union_rs_active;
    }

    bool is_both_now = (g_union_ag_active > 0) && (g_union_rs_active > 0);
    if (!was_both && is_both_now) g_union_both_start_ns = t;

    pthread_mutex_unlock(&g_union_lock);
}

void dpu_union_tracker_exit(bool is_ag) {
    uint64_t t = union_now_ns();
    pthread_mutex_lock(&g_union_lock);

    bool was_both = (g_union_ag_active > 0) && (g_union_rs_active > 0);

    if (is_ag) {
        g_union_ag_active--;
        if (g_union_ag_active == 0 && g_union_ag_busy_start_ns > 0) {
            g_union_ag_busy_total_ns += t - g_union_ag_busy_start_ns;
        }
    } else {
        g_union_rs_active--;
        if (g_union_rs_active == 0 && g_union_rs_busy_start_ns > 0) {
            g_union_rs_busy_total_ns += t - g_union_rs_busy_start_ns;
        }
    }

    bool is_both_now = (g_union_ag_active > 0) && (g_union_rs_active > 0);
    if (was_both && !is_both_now && g_union_both_start_ns > 0) {
        g_union_both_total_ns += t - g_union_both_start_ns;
    }

    pthread_mutex_unlock(&g_union_lock);
}

void dpu_union_tracker_print(int rank) {
    pthread_mutex_lock(&g_union_lock);
    if (g_union_printed || rank != 0) {
        pthread_mutex_unlock(&g_union_lock);
        return;
    }
    g_union_printed = 1;

    uint64_t ag_ns  = g_union_ag_busy_total_ns;
    uint64_t rs_ns  = g_union_rs_busy_total_ns;
    uint64_t both_ns = g_union_both_total_ns;
    uint64_t ag_cnt = g_union_ag_count;
    uint64_t rs_cnt = g_union_rs_count;
    uint64_t max_ag = g_union_max_ag_inflight;
    uint64_t max_rs = g_union_max_rs_inflight;
    pthread_mutex_unlock(&g_union_lock);

    double ag_s   = ag_ns / 1e9;
    double rs_s   = rs_ns / 1e9;
    double both_s = both_ns / 1e9;
    double any_s  = ag_s + rs_s - both_s;
    double ag_only_s = ag_s - both_s;
    double rs_only_s = rs_s - both_s;

    printf("\n========== [DPU UNION TRACKER rank=%d] (interval union / parallelism-aware) ==========\n"
           "  AG  busy (union): %8.3f s | ops=%lu | max inflight=%lu\n"
           "  RS  busy (union): %8.3f s | ops=%lu | max inflight=%lu\n"
           "  Both  (overlap) : %8.3f s  (AG and RS simultaneously active)\n"
           "  AG only         : %8.3f s  (AG active, RS idle)\n"
           "  RS only         : %8.3f s  (RS active, AG idle)\n"
           "  Any comm (union): %8.3f s  (= AG + RS − overlap)\n"
           "==========================================================================\n",
           rank,
           ag_s, ag_cnt, max_ag,
           rs_s, rs_cnt, max_rs,
           both_s, ag_only_s, rs_only_s, any_s);
    fflush(stdout);
}
