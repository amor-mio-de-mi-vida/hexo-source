---
date: 2024-11-03 21:29:41
date modified: 2024-11-03 23:15:58
title: Memory Hierarchies and Matrix Multiplication
tags:
  - course
categories:
  - cs267
---
## Memory Hierarchies

### Temporal and spatial locality

**Memory accesses (load / store) have two costs**

- Latency: cost to load or store 1 word ($\alpha$) $\approx$ delay due data travel time (secs)

- Bandwidth: Average rate (bytes/sec) to load/store a large chunk or inverse time/byte, $\beta$ $\approx$ data throughput (bytes/second)

reuse: spacial locality and temporal locality

### Basics of caches

- Cache is fast (expensive) memory which keeps copy of data; it is hidden from software.

- Cache hit: in-cache memory access —— cheap

- Cache miss: non-cached memory access —— expensive

**Associativity**

direct-mapped: only 1 address (line) in a given range in cache

- Data at address xxxxx1101 stored at cache location 1101

- Only 1 such value from memory

<mark>It isn't always better to add more levels of cache.</mark> too much cache miss ?

if your data doesn't fit in DRAM at all, that means it's going to have to go to disk. And to figure out where that is, there's all this sort of special hardware called the TLB to figure out where in memory that particular page lives.

**Approaches to Handling Memory Latency**

Reuse values in fast memory (bandwidth filtering)

- need temporal locality in program

Move larger chunks (achieve higher bandwidth)

- need spatial locality in program

Issue multiple reads/writes in single instruction (higher bw)

- vector operations require access set of locations (typically neighboring)

Issue multiple reads/writes in parallel (hide latency)

- prefetching issues read hint

- delayed writes (write buffering) stages writes for later operation

- <mark>both require that nothing dependent is happening (parallelism)</mark>

> Little's Law from queuing theory says
> *concurrency* = *latency* \* *bandwidth*

Key is finding enough parallelism in your code to know ahead of time what you need to load to go.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.9kg6w792ij.webp)

Given the complexity of the hardware, how do we abstract this to come up with just a simple model that a programmer can use without worrying about all these details? 

### Use of microbenchmarks to characterize performance

use benchmarks to infer the cache size.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.3d4sw2562g.webp)

Performance is complicated 

Memory is hierarchical

- Registers under compiler control

- Caches (multiple) under hardware control

- Order of magnitude between layers (speed and size)

Trends: growing gap

Little's Law: concurrency to overlap latency.

## Parallelism within single processors

### Instruction Level Parallelism (ILP) and Pipelining

pipelining doesn't change the latency, but it does improve the bandwidth significantly.

### SIMD units

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.5c0zmf3swe.webp)

Instructions perform add, multiply etc. on all values in parallel

Need to:

- Expose parallelism to compiler (or write manually)

- Be contiguous in memory and cache aligned (in most cases)

Parallelism can get the wrong answer if instructions execute out of order.

Types of dependencies

**Read-After-Write**

- `X=A; B=X;`

**Write-After-Read**

- `A=X; X=B;`

**Write-After-Write**

- `X=A; X=B;`

### Memory Alignment and Strides

- Non-contiguous memory access hampers efficiency

- Can be done if write conflicts are avoided

| type              | operation       |
| ----------------- | --------------- |
| strided load      | `...=a[i * 4]`  |
| strided store     | `a[i*4]=...`    |
| Indexed (gather)  | `... = a[b[i]]` |
| Indexed (scatter) | `a[b[i]]=...`   |

- Aligning on cache line boundaries key.

Understanding whether two locations would conflict can be very complicated.  Sometimes it's something that the user has to do.

### What does this mean to you?

In theory, the compiler understands all of this

- It will rearrange instructions to maximizes parallelism, uses FMAs and SIMD

- While preserving dependencies

But in practice the compiler may need your help

- Choose a different compiler, optimization flags, etc.

- Rearrange your code to make things more obvious

- Use special functions ("intrinsics") or write assembly

### Compiler Optimizations

- Unrolls loops (because control isn't free)

- Fuses loops (merge two together)

- Interchanges loops (reorder)

- Eliminates dead code (the branch never taken)

- Reorders instructions to improve register reuse and more

- Strength reduction (multiply by 2, into shift left)

- Strip-mine: turn one loop into nested one

- Strip-mine + interchange = tiling

![Survey by Prof. Susan Granham and students](https://dl.acm.org/doi/pdf/10.1145/197405.197406)

## Case study: Matrix Multiplication

### A Simple Model of Memory

Assume just 2 levels in hierarchy, fast and slow

All data initially in slow memory

- $m$ = number of memory elements (words) moved between fast and slow memory

- $t_m$ = time per slow memory operation (inverse bandwidth in best case)

- $f$ = number of arithmetic operations

- $t_f$ = time per arithmetic operation << $t_m$

- $CI$ = computational Intensity = $f/m$ = average number of flops per slow memory access

Minimum possible time = $f*t_f$ when all data in fast memory.

Actual time $f*t_f + m*t_m=f*t_f*(1+t_m/t_f * 1/CI)$

Larger $CI$ means time closer to minimum $f*t_f$

