#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_mmap.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_sync_event.h>
#include <doca_comch.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_pe.h>
#include <mpi.h>
#include "../common/comch_ctrl_path_common.h"
#include "../common/common.h"
#include "comch_server.h"
#include "../common/rdma_common.h"
#include "doca_utils_myself.h"

DOCA_LOG_REGISTER(DOCA_UTILS);

//ユーティリティ関数たち
doca_error_t config_init_with_device_name(struct rdma_config *cfg, const void *device_name_void, size_t device_name_len)
{
    size_t copy_len;
    if (cfg == NULL || device_name_void == NULL) {
        DOCA_LOG_ERR("Invalid argument in config_init_with_device_name");
        return DOCA_ERROR_INVALID_VALUE;
    }
    struct doca_log_backend *sdk_log;
    memset(cfg, 0, sizeof(*cfg));
    set_default_config_value(cfg);
    register_rdma_common_params();
    register_rdma_write_string_param();
    /* cfg->device_name に NUL 終端付きでコピー */
    copy_len = device_name_len;
    if (copy_len > sizeof(cfg->device_name) - 1)
        copy_len = sizeof(cfg->device_name) - 1;
    memcpy(cfg->device_name, device_name_void, copy_len);
    cfg->device_name[copy_len] = '\0';
    printf("device_name (cfg) = %s\n", cfg->device_name);
    return DOCA_SUCCESS;
}

void rdma_send_completed_callback(struct doca_rdma_task_send *rdma_send_task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    //いったんsendタスクをfreeするだけにします。
    DOCA_LOG_INFO("RDMA send task was done Successfully");
    struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
    resources->num_remaining_tasks--;
    doca_task_free(doca_rdma_task_send_as_task(rdma_send_task));
}

void rdma_send_error_callback(struct doca_rdma_task_send *rdma_send_task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    //いったんsendタスクをfreeするだけにします。
    struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	struct doca_task *task = doca_rdma_task_send_as_task(rdma_send_task);
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result;
	/* Update that an error was encountered */
	result = doca_task_get_status(task);
	DOCA_ERROR_PROPAGATE(*first_encountered_error, result);
	DOCA_LOG_ERR("RDMA receive task failed: %s", doca_error_get_descr(result));
	doca_task_free(task);
	result = doca_buf_dec_refcount(resources->dst_buf, NULL);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to decrease dst_buf count: %s", doca_error_get_descr(result));
}

void rdma_send_state_change_callback(const union doca_data user_data, struct doca_ctx *ctx, enum doca_ctx_states prev_state, enum doca_ctx_states next_state)
{
    struct rdma_resources *resources = (struct rdma_resources *)user_data.ptr;
	struct rdma_config *cfg = resources->cfg;
	doca_error_t result = DOCA_SUCCESS;
	(void)prev_state;
	(void)ctx;
	switch (next_state) {
	case DOCA_CTX_STATE_STARTING:
		DOCA_LOG_INFO("RDMA context entered starting state");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("RDMA context is running");
        break;
	case DOCA_CTX_STATE_STOPPING:
		DOCA_LOG_INFO("RDMA context entered into stopping state. Any inflight tasks will be flushed");
		break;
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("RDMA context has been stopped");
		resources->run_pe_progress = false;
		break;
	default:
		break;
	}
	if (result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(resources->first_encountered_error, result);
        DOCA_LOG_ERR("send state result is not DOCA_SUCCESS: %s", doca_error_get_descr(result));
		(void)doca_ctx_stop(ctx);
	}
}

void rdma_receive_completed_callback(struct doca_rdma_task_receive *rdma_receive_task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    //receiveタスクをfreeするだけにします。
    DOCA_LOG_INFO("RDMA receive task was done Successfully");
    struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
    resources->num_remaining_tasks--;
    doca_task_free(doca_rdma_task_receive_as_task(rdma_receive_task));
}

void rdma_receive_error_callback(struct doca_rdma_task_receive *rdma_receive_task, union doca_data task_user_data, union doca_data ctx_user_data)
{
    //receiveタスクをfreeするだけにします。
    struct rdma_resources *resources = (struct rdma_resources *)ctx_user_data.ptr;
	struct doca_task *task = doca_rdma_task_receive_as_task(rdma_receive_task);
	doca_error_t *first_encountered_error = (doca_error_t *)task_user_data.ptr;
	doca_error_t result;
	/* Update that an error was encountered */
	result = doca_task_get_status(task);
	DOCA_ERROR_PROPAGATE(*first_encountered_error, result);
	DOCA_LOG_ERR("RDMA receive task failed: %s", doca_error_get_descr(result));
	doca_task_free(task);
	result = doca_buf_dec_refcount(resources->dst_buf, NULL);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to decrease dst_buf count: %s", doca_error_get_descr(result));
}

void rdma_receive_state_change_callback(const union doca_data user_data, struct doca_ctx *ctx, enum doca_ctx_states prev_state, enum doca_ctx_states next_state)
{
	struct rdma_resources *resources = (struct rdma_resources *)user_data.ptr;
	struct rdma_config *cfg = resources->cfg;
	doca_error_t result = DOCA_SUCCESS;
	(void)prev_state;
	(void)ctx;

	switch (next_state) {
	case DOCA_CTX_STATE_STARTING:
		DOCA_LOG_INFO("RDMA context entered starting state");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("RDMA context is running");
        break;
	case DOCA_CTX_STATE_STOPPING:
		DOCA_LOG_INFO("RDMA context entered into stopping state. Any inflight tasks will be flushed");
		break;
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("RDMA context has been stopped");
		resources->run_pe_progress = false;
		break;
	default:
		break;
	}
	if (result != DOCA_SUCCESS) {
		DOCA_ERROR_PROPAGATE(resources->first_encountered_error, result);
        DOCA_LOG_ERR("recv state result is not DOCA_SUCCESS: %s", doca_error_get_descr(result));
		(void)doca_ctx_stop(ctx);
	}
}

doca_error_t create_rdma_resources(const void *device_name_void, size_t device_name_len, struct rdma_resources *resources, bool is_send, size_t working_buffer_len)
{
    if(resources == NULL || device_name_void == NULL) {
        DOCA_LOG_ERR("Invalid argument in create rdma send resources without mmap");
        return DOCA_ERROR_INVALID_VALUE;
    }

    struct rdma_config *cfg = NULL;
    cfg = calloc(1, sizeof(*cfg));
    if(cfg == NULL) {
        DOCA_LOG_ERR("Failed to allocate rdma config");
        return DOCA_ERROR_NO_MEMORY;
    }
    doca_error_t result = config_init_with_device_name(cfg, device_name_void, device_name_len);
    if(result != DOCA_SUCCESS) {
        free(cfg);
        return result;
    }
    resources->cfg = cfg;
    printf("after config init with device name\n");

    //ここからconfigを利用したRDMA資源の割り当て処理に入る
    //GPU接続の時はMMAPのパーミッションに "ACCESS FLAG PCIe" が必要かも
    //メモリマップ登録をしてね~~~
    if(is_send) {
        const uint32_t mmap_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
	    const uint32_t rdma_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
        DOCA_CHECK(allocate_rdma_resources_with_dynamic_memrange(cfg, mmap_permissions, rdma_permissions, doca_rdma_cap_task_send_is_supported, resources, working_buffer_len));
        //DOCA_CHECK(allocate_rdma_resources(cfg, mmap_permissions, rdma_permissions, doca_rdma_cap_task_send_is_supported, resources));
        DOCA_CHECK(doca_rdma_task_send_set_conf(resources->rdma, /*自前*/rdma_send_completed_callback, /*自前*/rdma_send_error_callback, /*1*/NUM_RDMA_TASKS));
        DOCA_CHECK(doca_ctx_set_state_changed_cb(resources->rdma_ctx, /*自前*/rdma_send_state_change_callback));
    } else {
        uint32_t mmap_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
	    uint32_t rdma_permissions = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE;
        DOCA_CHECK(allocate_rdma_resources_with_dynamic_memrange(cfg, mmap_permissions, rdma_permissions, doca_rdma_cap_task_receive_is_supported, resources, working_buffer_len));
        //DOCA_CHECK(allocate_rdma_resources(cfg, mmap_permissions, rdma_permissions, doca_rdma_cap_task_send_is_supported, resources));
        DOCA_CHECK(doca_rdma_task_receive_set_conf(resources->rdma, /*自前*/rdma_receive_completed_callback, /*自前*/rdma_receive_error_callback, /*1*/NUM_RDMA_TASKS));
        DOCA_CHECK(doca_ctx_set_state_changed_cb(resources->rdma_ctx, /*自前*/rdma_receive_state_change_callback));
    }

    return DOCA_SUCCESS;
}

doca_error_t rdma_ctx_set_user_data_and_start(struct rdma_resources *resources)
{
    if(resources == NULL || resources->rdma_ctx == NULL) {
        DOCA_LOG_ERR("Invalid argument in rdma_ctx_set_user_data_and_start");
        return DOCA_ERROR_INVALID_VALUE;
    }
    doca_error_t result;
    union doca_data ctx_user_data  = {0};
    ctx_user_data.ptr = resources;
    DOCA_CHECK(doca_ctx_set_user_data(resources->rdma_ctx, ctx_user_data));

    /* Create DOCA buffer inventory */
	result = doca_buf_inventory_create(INVENTORY_NUM_INITIAL_ELEMENTS, &resources->buf_inventory);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA buffer inventory: %s", doca_error_get_descr(result));
	}
	/* Start DOCA buffer inventory */
	result = doca_buf_inventory_start(resources->buf_inventory);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start DOCA buffer inventory: %s", doca_error_get_descr(result));
	}

    result = doca_ctx_start(resources->rdma_ctx);
    /* RDMA/UROM サンプル同様、IN_PROGRESS を正常扱いにする */
    if (result != DOCA_ERROR_IN_PROGRESS && result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("doca_ctx_start() failed: %s", doca_error_get_descr(result));
        return result;
    }
    /* コンテキストが RUNNING になるまで PE を progress する */
    enum doca_ctx_states state;
    do {
        doca_pe_progress(resources->pe);
        result = doca_ctx_get_state(resources->rdma_ctx, &state);
        printf("In doca pe progress state is %lu\n", state);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("doca_ctx_get_state() failed: %s", doca_error_get_descr(result));
            return result;
        }
    } while (state != DOCA_CTX_STATE_RUNNING);

    result = doca_ctx_get_state(resources->rdma_ctx, &state);
    return DOCA_SUCCESS;
}