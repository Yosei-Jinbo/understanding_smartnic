#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>

#include "doca_comch_client_interface.h"
#include "../common/comch_mpi_common.h"

namespace py = pybind11;

static py::object g_torch;
static std::string g_server_s;
static std::string g_pci_s;

static inline bool is_torch_tensor(const py::object& obj) {
    if (!g_torch) g_torch = py::module_::import("torch");
    return py::bool_(g_torch.attr("is_tensor")(obj));
}

/* obj から (addr,size,device) を取り出す（既存実装の踏襲） */
static void get_addr_size(py::object obj, uint64_t* addr, uint64_t* size, int* device) {
    if (!addr || !size || !device) {
        throw std::invalid_argument("addr/size/device pointers must not be null");
    }

    if (is_torch_tensor(obj)) {
        auto t = obj.cast<torch::Tensor>();
        if (!t.defined()) throw std::runtime_error("tensor is undefined");
        if (!t.is_contiguous()) throw std::runtime_error("tensor must be contiguous");

        *addr   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.data_ptr()));
        *size   = static_cast<uint64_t>(t.nbytes());
        *device = t.is_cuda() ? t.get_device() : -1;
        return;
    }

    if (!PyObject_CheckBuffer(obj.ptr())) {
        throw std::runtime_error("object is neither a torch.Tensor nor a CPU buffer");
    }

    py::buffer b = py::reinterpret_borrow<py::buffer>(obj);
    py::buffer_info info = b.request();

    *addr   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(info.ptr));
    *size   = static_cast<uint64_t>(info.size) * static_cast<uint64_t>(info.itemsize);
    *device = -1;
}

static void ucp_connect_host_dpu_request_py(uint64_t id)
{
    ucp_connect_host_dpu_request_c(id);
}

static void ucp_create_ring_request_py(uint64_t id)
{
    ucp_create_ring_request_c(id);
}

/* ===== 既存：blocking ===== */
static void ucp_collective_request_py(uint64_t id,
                                      py::object src_obj,
                                      py::object dst_obj,
                                      CollectiveCommunication collective_op,
                                      uint64_t root = static_cast<uint64_t>(-1))
{
    struct CollectiveRequest req;
    req.collective_op = collective_op;
    req.root = root;

    uint64_t src_addr, src_size, dst_addr, dst_size;
    int src_dev, dst_dev;
    get_addr_size(src_obj, &src_addr, &src_size, &src_dev);
    get_addr_size(dst_obj, &dst_addr, &dst_size, &dst_dev);

    {
        py::gil_scoped_release release;
        ucp_collective_request_c(id, src_addr, src_size, dst_addr, dst_size, req, 0, 0, 0, 0);
    }
}

/* ===== 追加：enqueue only（handle を返す） ===== */
static uint64_t ucp_collective_enqueue_py(uint64_t id,
                                          py::object src_obj,
                                          py::object dst_obj,
                                          CollectiveCommunication collective_op,
                                          uint64_t root = static_cast<uint64_t>(-1))
{
    struct CollectiveRequest req;
    req.collective_op = collective_op;
    req.root = root;

    uint64_t src_addr, src_size, dst_addr, dst_size;
    int src_dev, dst_dev;
    get_addr_size(src_obj, &src_addr, &src_size, &src_dev);
    get_addr_size(dst_obj, &dst_addr, &dst_size, &dst_dev);

    uint64_t handle = 0;
    int rc = 0;
    {
        py::gil_scoped_release release;
        rc = ucp_collective_enqueue_c(id, src_addr, src_size, dst_addr, dst_size, req, &handle, 0, 0, 0, 0);
    }
    if (rc < 0) {
        throw std::runtime_error("ucp_collective_enqueue_c failed: rc=" + std::to_string(rc));
    }
    return handle;
}

/* ===== Phase 14: enqueue with GPU flag (handle を返す) ===== */
static uint64_t ucp_collective_enqueue_with_flag_py(uint64_t id,
                                                    py::object src_obj,
                                                    py::object dst_obj,
                                                    CollectiveCommunication collective_op,
                                                    uint64_t flag_gpu_addr,
                                                    uint32_t flag_value,
                                                    uint64_t root = static_cast<uint64_t>(-1))
{
    struct CollectiveRequest req;
    req.collective_op = collective_op;
    req.root = root;

    uint64_t src_addr, src_size, dst_addr, dst_size;
    int src_dev, dst_dev;
    get_addr_size(src_obj, &src_addr, &src_size, &src_dev);
    get_addr_size(dst_obj, &dst_addr, &dst_size, &dst_dev);

    uint64_t handle = 0;
    int rc = 0;
    {
        py::gil_scoped_release release;
        rc = ucp_collective_enqueue_with_flag_c(id, src_addr, src_size, dst_addr, dst_size,
                                                req, &handle, 0, 0, 0, 0,
                                                flag_gpu_addr, flag_value);
    }
    if (rc < 0) {
        throw std::runtime_error("ucp_collective_enqueue_with_flag_c failed: rc=" + std::to_string(rc));
    }
    return handle;
}

/* ===== Phase 14: register the GPU flag pool with the host C side ===== */
static void register_flag_pool_py(uint64_t base_addr, uint64_t total_len)
{
    int rc = 0;
    {
        py::gil_scoped_release release;
        rc = comch_register_flag_pool_c(base_addr, total_len);
    }
    if (rc < 0) {
        throw std::runtime_error("comch_register_flag_pool_c failed: rc=" + std::to_string(rc));
    }
}

/* local_flag は int32_t の 1 要素テンソルとして渡し、ポインタとして扱う。 */
static void ucp_collective_local_cpu_request_py(uint64_t id,
                                      py::object src_obj,
                                      py::object dst_obj,
                                      py::object local_obj,
                                      py::object local_flag,
                                      CollectiveCommunication collective_op,
                                      uint64_t root = static_cast<uint64_t>(-1))
{
    struct CollectiveRequest req;
    req.collective_op = collective_op;
    req.root = root;

    uint64_t src_addr, src_size, dst_addr, dst_size, local_addr, local_size, local_flag_addr, local_flag_size;
    int src_dev, dst_dev, local_dev, local_flag_dev;
    get_addr_size(src_obj, &src_addr, &src_size, &src_dev);
    get_addr_size(dst_obj, &dst_addr, &dst_size, &dst_dev);
    get_addr_size(local_obj, &local_addr, &local_size, &local_dev);
    get_addr_size(local_flag, &local_flag_addr, &local_flag_size, &local_flag_dev);
    {
        py::gil_scoped_release release;
        ucp_collective_request_c(id, src_addr, src_size, dst_addr, dst_size, req, local_addr, local_size, local_flag_addr, local_flag_size);
    }
}

/* ===== 追加：enqueue only（handle を返す） ===== */
static uint64_t ucp_collective_local_cpu_enqueue_py(uint64_t id,
                                          py::object src_obj,
                                          py::object dst_obj,
                                          py::object local_obj,
                                          py::object local_flag, 
                                          CollectiveCommunication collective_op,
                                          uint64_t root = static_cast<uint64_t>(-1))
{
    struct CollectiveRequest req;
    req.collective_op = collective_op;
    req.root = root;

    uint64_t src_addr, src_size, dst_addr, dst_size, local_addr, local_size, local_flag_addr, local_flag_size;
    int src_dev, dst_dev, local_dev, local_flag_dev;
    get_addr_size(src_obj, &src_addr, &src_size, &src_dev);
    get_addr_size(dst_obj, &dst_addr, &dst_size, &dst_dev);
    get_addr_size(local_obj, &local_addr, &local_size, &local_dev);
    get_addr_size(local_flag, &local_flag_addr, &local_flag_size, &local_flag_dev);
    uint64_t handle = 0;
    int rc = 0;
    {
        py::gil_scoped_release release;
        rc = ucp_collective_enqueue_c(id, src_addr, src_size, dst_addr, dst_size, req, &handle, local_addr, local_size, local_flag_addr, local_flag_size);
    }
    if (rc < 0) {
        throw std::runtime_error("ucp_collective_enqueue_c failed: rc=" + std::to_string(rc));
    }
    return handle;
}

static bool comch_req_test_py(uint64_t handle)
{
    uint64_t result = 0;
    int rc = comch_req_test_c(handle, &result);
    if (rc < 0) {
        throw std::runtime_error("comch_req_test_c failed: rc=" + std::to_string(rc));
    }
    return (rc == 1);
}

static void comch_req_wait_py(uint64_t handle)
{
    uint64_t result = 0;
    int rc = 0;
    {
        py::gil_scoped_release release;
        rc = comch_req_wait_c(handle, &result);
    }
    if (rc < 0) {
        throw std::runtime_error("comch_req_wait_c failed: rc=" + std::to_string(rc));
    }
}

static void comch_req_release_py(uint64_t handle)
{
    comch_req_release_c(handle);
}

static void doca_comch_client_init_py(py::str server_name, py::str pci_addr)
{
    g_server_s = server_name.cast<std::string>();
    g_pci_s    = pci_addr.cast<std::string>();
    doca_comch_client_init_c(g_server_s.c_str(), g_pci_s.c_str());
}

static void doca_mpi_finalize_py()
{
    doca_mpi_finalize_c();
}

PYBIND11_MODULE(doca_comch_client_pybind, m) {
    py::enum_<CollectiveCommunication>(m, "CollectiveCommunication")
        .value("SIMPLE_RING",    CollectiveCommunication::COLLECTIVE_SIMPLE_RING)
        .value("REDUCE_SCATTER", CollectiveCommunication::COLLECTIVE_REDUCE_SCATTER)
        .value("ALL_GATHER",     CollectiveCommunication::COLLECTIVE_ALL_GATHER)
        .value("ALL_REDUCE",     CollectiveCommunication::COLLECTIVE_ALL_REDUCE)
        .value("BROADCAST",      CollectiveCommunication::COLLECTIVE_BROADCAST)
        .value("REDUCE",         CollectiveCommunication::COLLECTIVE_REDUCE)
        .export_values();

    m.def("ucp_connect_host_dpu_request_py", &ucp_connect_host_dpu_request_py, py::arg("id"));
    m.def("ucp_create_ring_request_py", &ucp_create_ring_request_py, py::arg("id"));

    /* blocking */
    m.def("ucp_collective_request_py", &ucp_collective_request_py,
          py::arg("id"), py::arg("src_obj"), py::arg("dst_obj"), py::arg("collective_op"),
          py::arg("root") = static_cast<uint64_t>(-1));

    /* async enqueue */
    m.def("ucp_collective_enqueue_py", &ucp_collective_enqueue_py,
          py::arg("id"), py::arg("src_obj"), py::arg("dst_obj"), py::arg("collective_op"),
          py::arg("root") = static_cast<uint64_t>(-1));

    /* Phase 14: async enqueue with GPU flag */
    m.def("ucp_collective_enqueue_with_flag_py", &ucp_collective_enqueue_with_flag_py,
          py::arg("id"), py::arg("src_obj"), py::arg("dst_obj"), py::arg("collective_op"),
          py::arg("flag_gpu_addr"), py::arg("flag_value"),
          py::arg("root") = static_cast<uint64_t>(-1));

    /* Phase 14: register GPU flag pool */
    m.def("register_flag_pool_py", &register_flag_pool_py,
          py::arg("base_addr"), py::arg("total_len"));

    m.def("ucp_collective_local_cpu_request_py", &ucp_collective_local_cpu_request_py,
          py::arg("id"), py::arg("src_obj"), py::arg("dst_obj"), py::arg("local_obj"), py::arg("local_flag"), py::arg("collective_op"),
          py::arg("root") = static_cast<uint64_t>(-1));

    /* async enqueue */
    m.def("ucp_collective_local_cpu_enqueue_py", &ucp_collective_local_cpu_enqueue_py,
          py::arg("id"), py::arg("src_obj"), py::arg("dst_obj"), py::arg("local_obj"), py::arg("local_flag"), py::arg("collective_op"),
          py::arg("root") = static_cast<uint64_t>(-1));

    m.def("comch_req_test_py", &comch_req_test_py, py::arg("handle"));
    m.def("comch_req_wait_py", &comch_req_wait_py, py::arg("handle"));
    m.def("comch_req_release_py", &comch_req_release_py, py::arg("handle"));

    m.def("doca_comch_client_init_py", &doca_comch_client_init_py,
          py::arg("server_name"), py::arg("pci_addr"));

    m.def("doca_mpi_finalize_py", &doca_mpi_finalize_py);
}