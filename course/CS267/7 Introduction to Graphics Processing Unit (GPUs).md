---
date: 2024-11-11 13:33:44
date modified: 2024-11-11 16:24:31
title: Introduction to Graphics Processing Unit (GPUs)
tags:
  - course
categories:
  - cs267
---
## Summary

GPUs gain efficiency from simpler cores and more parallelism

- Very wide SIMD (SIMT) for parallel arithmetic and latency-hiding

Heterogeneous programming with manual offload

- CPU to run OS, etc. GPU for compute

Massive (mostly data) parallelism required

- Not as strict as CPU-SIM (divergent addresses, instructions)

Threads in block share faster memory and barriers

- Blocks in kernel share slow device memory and atomics
<!-- more -->

## GPU

### What is a GPU?

{% note primary %}
**Hardware Big Idea \#1:** 

Remove components that help a single instruction stream run fast.

- Discover parallelism

- Consume energy
 {% endnote %}

{% note primary %}
**Hardware Idea \#2:** 

A larger number of (smaller simpler) cores. With this much parallelism, it's likely to be *data parallel* (same operation, different data)
 {% endnote %}
 
{% note primary %}
**Hardware Big Idea \#3:** 

Share the instruction stream
 {% endnote %}

NVIDIA will call this a `warp` of threads, or `SIMT`: single instruction multiple threads. 

The way they differentiate it from `SIMD` is in two ways.

- `SIMD` is usually a continuous vector, that you're operating on, and `SIMT` can be anywhere, just needs to be the same instruction. The data doesn't need to be living in the same location (memory coalescing)

- In `SIMT`, threads are alive continuously, whereas in the `SIMD` instruction, the default is to do scalar operations, and sometimes you go into the regime of doing vector operations. 
 
### What about Branches?

If you write your code without paying attention to this reality that threads within a so-called wrap, will be basically serialize the branches, then you can go into a place where only one  thread is active within the nested if statements.

{% note primary %}
**Hardware Idea \#4:** 

Masks
 {% endnote %}

> warp is the NVIDIAs concept of threads that execute the same instruction in one cycle. And if there's divergence, they're just gonna wait for both cases to execute.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.231w0o7w9v.png)

### Stalls

- Stalls: a core cannot run the next instruction because of a dependency on a previous one.

- Memory operations are 100s-1000s of cycles.

- Removed the fancy caches and prefetch logic that helps avoid stalls.

CPU have the fancy caches and prefetch logic, but GPUs don't care about the latency of this operation, but GPUs care about throughput, which means GPUs only care about how many different operation, how many total number of operations can be done in a given N unit of time.


{% note primary %}
**Hardware Idea \#5:** 

Use threads to hide high latency operations.
 {% endnote %}
 
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.8ojprpp3rg.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4n7qdc2eab.webp)

 https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf

## Programming GPUs

Running GPU Code (Kernel)

- Allocate memory on GPU

- Copy data to GPU

- Execute GPU program

- Wait to complete

- Copy results back to CPU

```cpp
// Example: Vector Addition (GPU Serial)
float* x = new float[N];
float* y = new float[N];

int size = N * sizeof(float);
float *d_x, *d_y; // device copies of x, y
cudaMalloc((void**)&d_x, size);
cudaMalloc((void**)&d_y, size);

cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

// Run kernel on GPU
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, d_x, d_y);

// Copy result back to host
cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
// Free memory
cudaFree(d_x); cudaFree(d_y);
delete[] x; delete[] y;
```
```cpp
// GPU function to add two vectors
__global__ void add(int n, float* x, float* y) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (index < n) {
		y[i] = x[i] + y[i];
	}
}
```

**Monolithic kernel**: assumes we have enough threads to more than cover the array size. There are hardware limits on numBlocks too (but they are high)

The only problem here is, it creates all these threads and keeps them alive. And that actually has certain costs in terms of scheduling.  It actually stops you from optimizing for those variables. 

```cpp
// Run kernel on GPU
int blockSize = 256;
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
add<<<32*numSMs, blockSize>>>(N, x, y);
```
```cpp
// GPU function to add two vectors
__global__ void add(int n, float* x, float* y) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}
```

**Grid-stride loops**: you can now limit the number of blocks to tune performance

When you limit the number of blocks in your grid, threads are reused for multiple computations, amortizes thread creation/destruction.

Grid is a collection of threads. Threads in a grid execute a kernel function and are divided into thread blocks.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4qrcb31i9s.webp)

`gridDim`: the total number of blocks launched by this kernel invocation, as declared when instantiating the kernel.

> **Why I'm limiting the number of threads within a block to 256?**
> 
> The deeper answer to this question is registers. It's a fixed resource that is very valuable. The more threads you launch, the more threads are competing for the register file. 
> 
> You might want to tune these variables so that your input fits, especially the ones that you repeatedly access, fits into registers much as possible.

### Grids and Thread Blocks

A 2D Grid of 2D Blocks

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4jo4fnrqms.webp)

`int blockId = blockIdx.x + blockIdx.y * gridDim.x;`

`int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;`


## Sharing and Synchronization

### Memory types on NVIDIA

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.67xhcusbic.webp)

- **Registers** per thread

- **Local** cached memory per thread

- **Shared** memory (shared in block)

- **Global** device level shared

- **Constant** cache shared by threads

- **Texture** cache shared by all block

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4n7qddzyx8.webp)

Use both blocks and threads -- Why?

`kernel_call<<<blocks, threads>>>`

**Limit on maximum number of threads/block**

- Threads alone won't work for large arrays

**Fast shared memory only between threads**

- Blocks alone are slower



Stencil Kernel
```cpp
__global__ void stencil_1d(int *in, int *out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + RADIUS;
	
	// Read input elements into shared memory
	temp[lindex] = in[gindex];
	
	if (threadIdx.x < RADIUS) { // fill in halos
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]
	}
	
	// Apply the stencil
	int result = 0;
	for (int offet = -RADIUS; offset <= RADIUS; offset++) {
		result += temp[lindex + offset];
	}
	
	// Store the result
	out[gindex] = result;
}
```

Problem: Race Condition! we need thread synchronization

- Synchronizes all threads within a block 

`void __syncthreads();`

- Used to prevent RAW / WAR / WAW hazards

- All threads in the block must reach the barrier

- If used inside a conditional, the condition must be uniform across the block.

> Note: An important things about all the barriers, is if you are trying to synchronize with a barrier, some group of threads, processors, anything, you have to make sure that the sync threads or barrier is called by **all the threads or processes.**

```cpp
__global__ void stencil-1d(int *in, int *out) {
	... 
	// setup temp halos...
	// Synchronize (ensure all the data is available)
	__syncthreads();
	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		result += temp[lindex + offset];
	}
	// Store the result
	out[gindex] = result;
}
```

- Threads within a block may synchronize with barriers

- Blocks coordinate via atomic memory operations

- Implicit barrier between kernels

### Blocks must be independent

**Any possible interleaving of blocks should be valid**

- presumed to run to completion without pre-emption (not fairly scheduled)

- can run in any order

- can run concurrently OR sequentially

**Blocks may coordinate but not synchronize**

- shared queue pointer: **OK**

- shared lock: **BAD** ... can easily deadlock

**Independence requirement gives scalability**

### Mapping CUDA to NVIDIA GPUs

Threads:

- Each thread is a SIMD lane (ALU)

Warps:

- A warp executed as a logical SIMD instruction (sort of)

- Warp width is 32 elements: **LOGICAL** SIMD width

- (Warp-level programming also possible)

Thread blocks:

- Each thread block is scheduled onto an SM

- Peak efficiency requires multiple thread blocks per processor

Kernel

- Executes on a GPU (there is also multi-GPU programming)

### Memory coalescing

When you're fetching data from global memory to somewhere, for execution, You don't even need to copy it from shared memory. You can be fetching it for execution to registers. 

Successive 4W bytes (W: warp size, 4: size of single word in bytes) memory can be accessed by a warp (W consecutive threads) in a single transaction.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.26lhyivgg6.webp)

The following conditions may result in **uncoalesced** load, i.e., memory access becomes **serialized**: 

- memory is not **sequential**

- memory access is **sparse** (some addressed skipped)

- **misaligned** memory access

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.8ada0z2tu3.webp)



