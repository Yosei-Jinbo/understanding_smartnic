//完了キュー・通知キューの実装: 異なるコア上に配置されているスレッド間の通信を高速に行う
#ifndef SPSC_QUEUE_H
#define SPSC_QUEUE_H

#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "comch_server.h" //doca_rdma_task_type_t, doca_task_descのインクルード

//キャッシュ来の境界
#define CACHE_LINE_SIZE 64

//完了通知の構造体
typedef enum {
    COMP_SUCCESS,
    COMP_ERROR,
    COMP_CANCELLED,
} completion_status_t;

typedef struct {
    completion_status_t status;
    int stride_id; //リクエスト側にどのストライド分が終了したのかを伝える
} completion_event_t;

/* =====================================================
 * SPSC（Single Producer Single Consumer）キュー マクロテンプレート
 *
 * 使い方:
 *   DEFINE_SPSC_QUEUE(spsc_doca_task_queue, struct doca_task_desc, 64)
 * で以下が生成される:
 *   - 型:   spsc_doca_task_queue_t
 *   - 関数: spsc_doca_task_queue_init / _push / _pop / _is_empty
 * ===================================================== */

#define DEFINE_SPSC_QUEUE(NAME, ELEM_TYPE, CAPACITY)                          \
                                                                              \
typedef struct {                                                              \
    ELEM_TYPE buf[CAPACITY];                                                  \
    _Alignas(CACHE_LINE_SIZE) atomic_uint head;                               \
    char _pad1[CACHE_LINE_SIZE - sizeof(atomic_uint)];                        \
    _Alignas(CACHE_LINE_SIZE) atomic_uint tail;                               \
    char _pad2[CACHE_LINE_SIZE - sizeof(atomic_uint)];                        \
} NAME##_t;                                                                   \
                                                                              \
static inline void NAME##_init(NAME##_t *q)                                   \
{                                                                             \
    atomic_init(&q->head, 0);                                                 \
    atomic_init(&q->tail, 0);                                                 \
}                                                                             \
                                                                              \
static inline bool NAME##_push(NAME##_t *q, const ELEM_TYPE *elem)            \
{                                                                             \
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_relaxed);     \
    unsigned head = atomic_load_explicit(&q->head, memory_order_acquire);     \
    unsigned next = (tail + 1) & ((CAPACITY) - 1);                            \
                                                                              \
    if (next == head)                                                         \
        return false; /* キューがfull */                                      \
                                                                              \
    q->buf[tail] = *elem;                                                     \
    atomic_store_explicit(&q->tail, next, memory_order_release);              \
    return true;                                                              \
}                                                                             \
                                                                              \
static inline bool NAME##_pop(NAME##_t *q, ELEM_TYPE *elem)                   \
{                                                                             \
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);     \
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_acquire);     \
                                                                              \
    if (head == tail)                                                         \
        return false; /* キューが空 */                                        \
                                                                              \
    *elem = q->buf[head];                                                     \
    atomic_store_explicit(&q->head, (head + 1) & ((CAPACITY) - 1),           \
                          memory_order_release);                               \
    return true;                                                              \
}                                                                             \
                                                                              \
static inline bool NAME##_is_empty(NAME##_t *q)                               \
{                                                                             \
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);     \
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_acquire);     \
    return head == tail;                                                      \
}

/* =====================================================
 * キューのインスタンス化
 * ===================================================== */

/* DOCA RDMA タスクキュー (64 スロット) */
DEFINE_SPSC_QUEUE(spsc_doca_task_queue,    struct doca_task_desc, 64)

/* 完了通知キュー (64 スロット) */
DEFINE_SPSC_QUEUE(spsc_completion_queue,   completion_event_t,   64)

/* =====================================================
 * RS (Reduce-Scatter) タスク用SPSCキュー
 * 集約演算のsubmit/wait分離を実現
 * ===================================================== */

struct rs_task {
    fp16_t   *dst;
    fp16_t   *src;
    size_t    start;
    size_t    end;
    int       stride_id;    /* 完了追跡用ID（UCP_taskと同様） */
};

/* RSタスクキュー (64 スロット) */
DEFINE_SPSC_QUEUE(spsc_rs_task_queue,      struct rs_task,       64)

/* RS完了通知キュー (64 スロット) */
DEFINE_SPSC_QUEUE(spsc_rs_completion_queue, completion_event_t,  64)

#endif /* SPSC_QUEUE_H */
