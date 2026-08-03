#define _GNU_SOURCE
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <doca_comch.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <mpi.h>
#include "uthash.h"
#include <pthread.h>

#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "comch_server.h"
#include "../common/comch_mpi_common.h"
#include "../common/timing_utils.h"
#include "../common/doca_rdma_utils.h"
#include <sched.h>
#include <math.h>
#include "spsc_queue.h"

#include "rs_pool.h"
#include "dpu_config.h"

DOCA_LOG_REGISTER(RS_POOL);

/* * 集約用スレッドプール (rs_thread_pool) */

struct rs_thread_pool_t {
    int                      num_threads;
    pthread_t               *threads;
    struct rs_worker_arg_t  *worker_args;
    atomic_bool              running;
    spsc_rs_task_queue_t    *task_queue;
    spsc_rs_completion_queue_t *completion_queue;
    pthread_mutex_t          barrier_mutex;
    pthread_cond_t           barrier_cond;
    struct rs_task           current_task;
    atomic_int               done_count;
    atomic_int               ready_count;
    volatile int             task_generation;  /* busy-poll 用に volatile */
};

struct rs_worker_arg_t {
    struct rs_thread_pool_t *pool;
    int                      tid;
};

static void *
rs_worker_main(void *arg)
{
    struct rs_worker_arg_t  *wa   = (struct rs_worker_arg_t *)arg;
    struct rs_thread_pool_t *pool = wa->pool;
    const int tid = wa->tid;
    const int num_threads = pool->num_threads;
    int my_generation = 0;

    while (atomic_load(&pool->running)) {
        if (tid == 0) {
            struct rs_task task;
            pthread_mutex_lock(&pool->barrier_mutex);
            while (!spsc_rs_task_queue_pop(pool->task_queue, &task) && atomic_load(&pool->running))
                pthread_cond_wait(&pool->barrier_cond, &pool->barrier_mutex);
            if (!atomic_load(&pool->running)) { pthread_mutex_unlock(&pool->barrier_mutex); break; }
            pool->current_task = task;
            pool->task_generation++;
            atomic_store(&pool->done_count, 0);
            atomic_store(&pool->ready_count, 0);
            pthread_cond_broadcast(&pool->barrier_cond);
            pthread_mutex_unlock(&pool->barrier_mutex);
            my_generation = pool->task_generation;
        } else {
            pthread_mutex_lock(&pool->barrier_mutex);
            while (pool->task_generation == my_generation && atomic_load(&pool->running))
                pthread_cond_wait(&pool->barrier_cond, &pool->barrier_mutex);
            my_generation = pool->task_generation;
            pthread_mutex_unlock(&pool->barrier_mutex);
            if (!atomic_load(&pool->running)) break;
        }

        struct rs_task *task = &pool->current_task;
        uint64_t total = task->end - task->start;
        uint64_t base  = total / (uint64_t)num_threads;
        uint64_t rem   = total % (uint64_t)num_threads;
        uint64_t my_start = task->start + (uint64_t)tid * base + (uint64_t)((tid < (int)rem) ? tid : (int)rem);
        uint64_t my_len   = base + ((tid < (int)rem) ? 1ULL : 0ULL);
        uint64_t my_end   = my_start + my_len;

        if (my_len != 0)
            rs_add_range_neon(task->dst, task->src, my_start, my_end);

        int count = atomic_fetch_add(&pool->done_count, 1) + 1;
        if (count == num_threads) {
            completion_event_t event = { .status = COMP_SUCCESS, .stride_id = task->stride_id };
            uint64_t push_start_ns = now_monotonic_ns();
            while (!spsc_rs_completion_queue_push(pool->completion_queue, &event)) {
                uint64_t elapsed = now_monotonic_ns() - push_start_ns;
                if (elapsed < 200000) cpu_relax(); else sched_yield();
            }
        }
        atomic_fetch_add(&pool->ready_count, 1);
        if (tid == 0) {
            while (atomic_load(&pool->ready_count) < num_threads) cpu_relax();
        }
    }
    return NULL;
}

doca_error_t
rs_thread_pool_create(struct rs_thread_pool_t **out_pool, int num_threads)
{
    if (num_threads <= 0) num_threads = RS_NUM_THREADS;
    struct rs_thread_pool_t *pool = (struct rs_thread_pool_t *)calloc(1, sizeof(*pool));
    if (!pool) return DOCA_ERROR_NO_MEMORY;

    pool->num_threads = num_threads;
    pool->threads = (pthread_t *)calloc(num_threads, sizeof(pthread_t));
    pool->worker_args = (struct rs_worker_arg_t *)calloc(num_threads, sizeof(struct rs_worker_arg_t));
    pool->task_queue = (spsc_rs_task_queue_t *)calloc(1, sizeof(spsc_rs_task_queue_t));
    pool->completion_queue = (spsc_rs_completion_queue_t *)calloc(1, sizeof(spsc_rs_completion_queue_t));
    if (!pool->threads || !pool->worker_args || !pool->task_queue || !pool->completion_queue) {
        free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
        return DOCA_ERROR_NO_MEMORY;
    }
    spsc_rs_task_queue_init(pool->task_queue);
    spsc_rs_completion_queue_init(pool->completion_queue);
    pthread_mutex_init(&pool->barrier_mutex, NULL);
    pthread_cond_init(&pool->barrier_cond, NULL);
    pool->task_generation = 0;
    atomic_init(&pool->done_count, 0);
    atomic_init(&pool->ready_count, 0);
    atomic_init(&pool->running, true);

    for (int i = 0; i < num_threads; i++) {
        pool->worker_args[i].pool = pool;
        pool->worker_args[i].tid  = i;
        int rc = pthread_create(&pool->threads[i], NULL, rs_worker_main, &pool->worker_args[i]);
        if (rc != 0) {
            atomic_store(&pool->running, false);
            pthread_cond_broadcast(&pool->barrier_cond);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->barrier_mutex);
            pthread_cond_destroy(&pool->barrier_cond);
            free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
            return DOCA_ERROR_NO_MEMORY;
        }
        cpu_set_t cpuset; CPU_ZERO(&cpuset);
        CPU_SET(compute_core(i), &cpuset);
        pthread_setaffinity_np(pool->threads[i], sizeof(cpu_set_t), &cpuset);
    }
    *out_pool = pool;
    return DOCA_SUCCESS;
}

void rs_thread_pool_destroy(struct rs_thread_pool_t *pool)
{
    if (!pool) return;
    pthread_mutex_lock(&pool->barrier_mutex);
    atomic_store(&pool->running, false);
    pthread_cond_broadcast(&pool->barrier_cond);
    pthread_mutex_unlock(&pool->barrier_mutex);
    for (int i = 0; i < pool->num_threads; i++) pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->barrier_mutex);
    pthread_cond_destroy(&pool->barrier_cond);
    free(pool->threads); free(pool->worker_args); free(pool->task_queue); free(pool->completion_queue); free(pool);
}

void
submit_rs_task(struct rs_thread_pool_t *pool, fp16_t *dst, fp16_t *src, size_t start, size_t end, int stride_id)
{
    if (!pool || start >= end) return;
    struct rs_task task = { .dst = dst, .src = src, .start = start, .end = end, .stride_id = stride_id };
    while (!spsc_rs_task_queue_push(pool->task_queue, &task)) cpu_relax();
    pthread_mutex_lock(&pool->barrier_mutex);
    pthread_cond_broadcast(&pool->barrier_cond);
    pthread_mutex_unlock(&pool->barrier_mutex);
}

void
poll_wait_rs_completion(struct rs_thread_pool_t *pool, int stride_id)
{
    if (!pool) return;
    completion_event_t event;
    uint64_t start_ns = now_monotonic_ns();
    while (1) {
        if (spsc_rs_completion_queue_pop(pool->completion_queue, &event))
            if (event.stride_id == stride_id) return;
        uint64_t elapsed = now_monotonic_ns() - start_ns;
        if (elapsed < 200000) cpu_relax(); else sched_yield();
    }
}
