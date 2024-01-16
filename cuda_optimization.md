# CUDA Optimzation Part 1
The goal of CUDA programs is partially to use a lot of threads, i.e. mass parallelization.

## General Architecture
- SM stands for streaming multiprocessor. 
- GPU architectures are largely defined by their SM.
- The SM defines the instruction set.
- Large GPUs have more SMs on the die, smaller GPUs have fewer SMs on the die.
- Kepler GPUs have 192 SP units aka "cores" per SM.
- SP units are designed to do single precision floating point operations.
- Kepler GPUs have 64 DP units per SM.
- DP Units are designed to do double precision floating point operations.
- Each SM has warp schedulers which issue instructions.
- GPUs within a generation differ by the number of SMs and memory.
- Pascal/Volta GPUs have FP16 operations at 2x the SP rate.
- Compute capability 7.0 introduced tensor cores.
- Tensor cores are instructions optimized to perform matrix-matrix multiplication.
- CUDA kernels can be specified to use more cores than physically present, this allows them to expand to make use of newer cards with a greater number of cores.
- SP, DP and tensor core units can be used simultaneously.

## Execution Model
- CUDA programs consist of threads which are grouped into thread blocks which are grouped into grids.
- Thread -> thread block -> grid.
- On the hardware side, threads are executed by scalar processors, thread blocks are executed on multiprocessors, concurrent thread blocks reside on a single multiprocessor (SM), and a kernel is launched as a grid of thread blocks.
- Scalar processor -> multiprocessor (SM), device.
- Thread blocks stay on the same SM for their entire lifetime.
- Since shared memory is built into each SM, and all threads of a thread block execute on the same SM, all threads of a thread block can access the same shared memory.

## Warps
- A warp is a collection of 32 threads.
- A thread block can be decomposed into one or more warps.
- A thread block with 32 threads can be decomposed into one warp.
- GPUs issue instructions to warps, each thread within a warp then executes physically in parallel.

## Launch Configuration
- Instructions are issued in order.
- A thread stalls when one of the operands isn't ready, i.e. given multiplication of a and b, both a and b must be available at the time of the instruction or else the thread will stall.
- Latency is hidden by switching threads.
- Everything done inside a GPU has a latency associated with it.
- GMEM latency (global memory latency) is probably the largest latency that has to be dealt with.
- How many threads/threadblocks should be launched? Enough threads to hide latency.

## GPU Latency Hiding
- CUDA C source code:
```cuda
int idx = threadIdx.x + blockDim.x * blockIdx.x;
c[idx] = a[idx] * b[idx];
```
- Machine code (SASS/streaming assembler):
```
I0: LD R0, a[idx];      // Load a[idx] into R0
I1: LD R1, b[idx];      // Load b[idx] into R1
I2: MPY R2,R0,R1        // Multiply R0 and R1, and store the result in R2
```
- PTX is an intermediate code (inbetween CUDA and PTX) similar to LLVM IR.
- Objective is to keep the machine busy. Idle time means decreased performance.
- The GPU is an in order machine meaning each warp must execute I0 before I1 and I1 before I2.
- I0 and I1 are both read instructions so they can both be issued to a warp without stalling it.
- I2 must wait for a[idx] and b[idx] to populate R0 and R1 before it can execute.
- Therefore if I2 is executed before the memory requested from I0 and I1 arrives, the warp will stall.
- One strategy for reducing latency is to issue I0 and I1 one warp, then the next, and so on until R0 and R1 are populated in the first warp, at which point I2 can be issued on the first warp.
- The warp scheduler can hide latency by issuing instructions to unblocked warps on each cycle, i.e. issue I0 and I1 to unblocked warps while it waits for the memory to be loaded on other warps, after which it can issue I2 on those warps.
- By using more warps/threads, more latency is hidden.
- This type of latency hiding is done without the use of caches, but caches can also help by reducing the time it takes to retrive data from memory.
- Dual issue warp schedulers can hide latency by issuing two independent instructions at the same time.
- There is a relationship between number of threads and potential global memory throughput.
- Using more threads makes it easier to maximize global memory throughput.
- Operating on larger chunks of data per thread also makes it easier to maximize global memory throughput.
- Saturating the memory bus is important because global memory access is SLOW.

## What is Occupancy?
- A measure of the actual thread load in an SM vs peak theoretical/peak achievable load.
- Achievable occupancy is affected by limiters to occupancy.
- Primary limiters are registers per thread, threads per threadblock, and shared memory usage.

## Summary
- Use as many threads as possible to hide latency and maximize performance.
- Latency is not the only limit on performance.
- Thread blocks should be a multiple of the warp size.
- Instructions are issued warp-wide, each warp has 32 threads, so if thread blocks are not a multiple of warp size, there will be warps that have occupied but inactive threads and LD/ST units.
- If thread blocks are a multiple of warp size then it can be ensured that all the threads are being used.
- SMs can concurrently execute at least 16 thread blocks (Maxwell/Pascal/Volta).
- Max number of threads per thread block is 1024, so multiple thread blocks can be used in order to use more threads.
- Very large and vary small thread blocks can make it difficult to achieve good occupancy.
