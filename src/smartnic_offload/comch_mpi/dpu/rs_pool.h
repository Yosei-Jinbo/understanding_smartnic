#ifndef RS_POOL_H
#define RS_POOL_H

#include "comch_server.h"

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

/* RS (Reduce-Scatter) 集約用スレッドプール (rs_pool.c) */

/* RS_NUM_THREADS: フォールバック用デフォルト。実際の計算スレッド数は
 * 実行時 g_compute_cores（env COMPUTE_CORES）を使用。ピン留めは compute_core(i)（rank 共有）。 */
#define RS_NUM_THREADS           COMPUTE_CORES_DEFAULT
#define RS_AGGREGATION_THRESHOLD (256 * 1024)

/* fp16 加算。小サイズ RS はスレッドプールを使わず呼び出し元で直接実行する。 */
static inline void
rs_add_range_neon(fp16_t *dst, fp16_t *src, uint64_t start, uint64_t end)
{
    (void)(end - start);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint64_t vec_len  = len & ~7ULL;
    uint64_t i        = start;
    uint64_t loop_end = start + vec_len;
    for (; i < loop_end; i += 8) {
        float16x8_t vec_a = vld1q_f16((const __fp16 *)&dst[i]);
        float16x8_t vec_b = vld1q_f16((const __fp16 *)&src[i]);
        float16x8_t vec_r = vaddq_f16(vec_a, vec_b);
        vst1q_f16((__fp16 *)&dst[i], vec_r);
    }
    for (; i < end; i++)
        dst[i] = (fp16_t)(dst[i] + src[i]);
#else
    for (uint64_t i = start; i < end; i++)
        dst[i] = (fp16_t)(dst[i] + src[i]);
#endif
}

doca_error_t rs_thread_pool_create(struct rs_thread_pool_t **out_pool, int num_threads);
void rs_thread_pool_destroy(struct rs_thread_pool_t *pool);
void submit_rs_task(struct rs_thread_pool_t *pool, fp16_t *dst, fp16_t *src, size_t start, size_t end, int stride_id);
void poll_wait_rs_completion(struct rs_thread_pool_t *pool, int stride_id);

#endif /* RS_POOL_H */
