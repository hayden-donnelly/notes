# CUDA Optimzation Part 1
[Video](https://www.youtube.com/watch?v=cXpTDKjjKZE)

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

# Cuda Optimization Part 2
[Video](https://www.youtube.com/watch?v=Uz3r_OGQaxc)

## Memory Hierarchy Review
- Each thread has its own local storage, typically these are registers.
- Registers are generally managed by the compiler.
- Registers are FAST.
- Shared memory is accessible by threads in the same threadblock.
- L1 cache retains recently used data so that it can be accessed quickly if it is need again.
- L1 cache and shared memory have very low latency because they are both on the GPU chip.
- L2 cache is a device wide resource, all accesses to global memory go through L2, this includes copes to/from the host.
- Global memory is accessible by all threads as well as by the host.
- Global memory has a latency of hundreds of cycles.
- Host (CPU, Chipset, DRAM) -> (Device -> DRAM -> GPU (L1/L2 cache -> Multiprocessor)).
- There are other caches on the GPU which may be useful.
- Global device memory (DRAM) is not on the GPU chip itself while the L1 and L2 caches are.\
- It is on the overall GPU board though, i.e. the thing containing all parts of the GPU, similar to how a motherboard hosts a CPU and RAM but the RAM is not on the CPU chip itself.

## GMEM (Global Memory) Operations
- When loading data, first the L1 cache will be checked to see if a copy can be loaded from there, if it can't then the L2 cache will be checked for the same purpose, and if it still can't then the data will be loaded from global memory.
- The smallest unit of data that can be loaded from L1/L2 cache is a 128-byte line.
- Store operations invalidate the L1, and "write-back" data to L2 which eventually writes it to gmem.
- Non-caching loads skip the L1 cache and go straight to L2, the granularity of this type of load is 32-bytes.
- Non-caching loads invalidate data if it happens to be in the L1.
- In some corner cases, non-caching loads will give better performance.

## Load Operation
- Memory operations are issued per warp (i.e. 32 threads at a time).
- Every instruction on the GPU is issued per warp.
- A "line" is a subdivision of a cache.
- In CUDA GPUs the "unit of transaction" for memory is a segment which is 32 bytes.
- This means lines/segments are requested from memory, not individual bytes.

## Caching Load
- Since gmem loads work in 32 byte segments, it is important that the requested data makes use of all 32 bytes or the memory bus utilization will not be 100%.
- If 48 bytes of data are requested from global memory, then two 32 byte segments will be sent over the memory bus, therefore wasting 16 bytes.
- On the other hand, if a warp requests 32 aligned, consecutive 4-byte words then the addresses fall within 1 cache line, i.e. 4 segments.
- In this case the bus utilization is 100% and no bytes are wasted.
- This is called "perfect coalescing".
- The process of grouping memory into lines or segments is called coalescing.
- Requesting memory partially on both sides of a line or segment boundary is bad because the addresses will be coalesced into two groups on either side of the boundary, therefore the memory controller will have to retrieve two lines/segments which are not entirely required and the bus utilization will drop.
- A silver lining of this is that other warps may be able to use the previously unused data, and if this happens then the access will be much quicker because the data is already in cache.
- Worst case pattern is randomly accessing a few bytes of memory across many different segments, i.e. 4 bytes from segment 1, 4 bytes from segment 2, 4 bytes from segment 3, etc.
- In this case, most of the segment bytes are wasted and the bus utilization is very low.

## Non-Caching Load Revisited
- Non-caching loads from gmem can improve bus utilization in certain cases because we don't have to worry about utilizing 128 byte cache lines, and instead only need to be concerned with using 32 byte segments.

# GMEM Optimization Guidelines
- Strive for perfect coalescing, for example by aliging starting addresses to segment boundaries and programming warps to access contiguous regions in memory.
- Have enough concurrent memory accesses to saturate the bus, for example by launching enough threads and processing several elements per thread.
- Make use of all the caches (L1 and L2 are not explicitly user managed, but there other caches that are, i.e. constant cache and read-only cache).

## Shared Memory
- Not backed by off chip DRAM, it is on the GPU chip itself.
- Per SM resource, not global.
- Can be used for inter-thread communication.
- Memory resources that are closer to the processor are faster (shared memory is faster than global memory).
- Can be used to improve global memory access patterns.
- Can think of this as a 2-dimensional array of memory.
- 32 banks, 4 byte wide banks.
- Rows are 32 banks with 4 bytes each, each column represents a bank, so position (0, 0) is bank 1 with the first 4 bytes, positino (1, 0) is bank 1 with the second 4 bytes, and so on.
- Shared memory can deliver one item from each bank per cycle.
- Shared accessed are issued per warp.
- Accessing 4 byte words from different banks are executed in parallel.
- Accessing different 4 byte words from the same banks are executed serially.
- Multiple threads can access the same word from a bank in parallel, this is called multicast.
- In shared memory we refer to bank conflicts instead of coalescing.
- For best performance, avoid bank conflicts (reading/writing to different words in the same bank).
- When there are at most 2 threads accessing the same bank, this is called a 2-way bank conflict and it runs 2x slower than the best case.
- When there are at most 16 threads accessing the same bank, this is called a 16-way bank conflict and it runs 16x slower than the best case.
- Shared memory is often abbreviated as smem or shmem.
- Padding can help avoid bank conflicts by staggering columns across multiple banks.

## Summary
- Maximize global memory throughput (aim for 100% bus utilization).
- Use shared memory when possible.
- Use analysis/profiling for optimization.
