# Understanding Interference-Overhead Trade-offs of SmartNIC-Based Collective Communication Offloading for Distributed Training

Training, particularly for large-scale models, has increasingly relied on multiple GPUs and, more recently, heterogeneous devices such as CPUs. However, these approaches introduce substantial communication overhead to synchronize model states. Although overlapping communication with computation on GPUs is a common technique to hide this overhead, its effectiveness remains limited because communication and computation kernels contend for GPU resources. SmartNICs, as alternative computing devices to GPUs and CPUs, have been expected to relax this resource contention by offloading communication tasks.

The recent deployment of SoC SmartNICs in commercial AI data centers motivates to revisit SmartNIC offloading for distributed training, with a particular focus on their general-purpose cores and memory subsystems, which are slower than those of GPUs. We empirically evaluate whether SoC SmartNICs can accelerate distributed training. To this end, we carefully design and implement a reference SmartNIC offloading mechanism that incorporates multicore progress control, chunk-based pipelining, and direct device-to-device access to reduce SmartNIC-side overhead.

Our evaluation with NVIDIA BlueField 3 SmartNICs shows that SmartNIC offloading is effective when GPU-side communication interference is the dominant bottleneck. For ViT-L/16, offloading AllGather reduces end-to-end training step time by up to 30% and shortens the elapsed time of the forward and backward passes by 26% by reducing GPU waiting time. However, the benefit diminishes for communication-heavy workloads such as OPT-1.3B, where SmartNIC-side communication overhead becomes dominant. We further find that offloading ReduceScatter can be counterproductive because reductions consume limited SmartNIC resources. These results suggest that SoC SmartNIC offloading should prioritize communication-control-dominated collectives such as AllGather while leaving reduction-heavy collectives on the GPU.

## Hardware Requirements

This work targets SoC SmartNICs. Specifically, our reference implementation requires NVIDIA **BlueField-3 DPU** SmartNICs on each host. The evaluation setup uses two hosts (`bluefield01`, `bluefield02`), each equipped with a BlueField-3 DPU and a GPU.

## Build

The SmartNIC offloading mechanism consists of a host-side component (loaded by the training process) and a DPU-side component (running on the BlueField). Both must be built before launching the experiments.

### Host side

```bash
cd src/smartnic_offload/comch_mpi/host
USE_MARCH_NATIVE=1 USE_LTO=1 python3 setup.py build_ext --inplace
```

### DPU side

Run on the BlueField DPU:

```bash
cd src/smartnic_offload/comch_mpi/dpu
./build.sh
```

## Run

### SmartNIC offloading

Launch the DPU-side process **first**, then the host-side process. The host side will block until it can connect to the DPU side, so the order matters.

1. On the BlueField DPU:

   ```bash
   cd src/smartnic_offload/comch_mpi/dpu
   mpirun --app dpu_appfile
   ```

2. On the host:

   ```bash
   cd scripts/smartnic_offload
   mpirun --app host_appfile_<model>_<method>[_nsys]
   ```

   - `<model>` = `deberta_xl` | `opt_1.3b` | `vit_l_16`
   - `<method>` = `ag_smartnic` (AllGather offloading)
   - Append `_nsys` to capture an Nsight Systems profile (for example, `host_appfile_vit_l_16_ag_smartnic_nsys`); omit it for a normal run.

### Baseline (CPU buffering)

The baseline does not use the SmartNIC, so only the two hosts need to run. Launch the script with the node rank on **each** host.

```bash
# On bluefield01 (rank 0)
cd scripts/baseline
./run_<model>_prefetch[_nsys].sh 0

# On bluefield02 (rank 1)
cd scripts/baseline
./run_<model>_prefetch[_nsys].sh 1
```

- `<model>` = `deberta_xl` | `opt_1.3b` | `vit_l_16`
- Append `_nsys` to enable the Nsight Systems profile; omit it for a normal run.
