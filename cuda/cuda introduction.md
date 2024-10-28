---
title: cuda introduction
tags:
  - cuda
categories:
  - NVIDIA cuda 编程指南
date: 2024-10-16 09:23:28
date modified: 2024-10-19 17:29:18
---
## cuda 介绍
GPU 特别适合于并行数据运算的问题－同一个程序在许多并行数据元素，并带有高运算密度（算术运算与内存操作的比例）。由于同一个程序要执行每个数据元素，降低了对复杂的流量控制要求; 并且，因为它执行许多数据元素并且据有高运算密度，内存访问的延迟可以被忽略。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.4qrb9lvfly.webp)

cuda软件堆栈由以下几层组成

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.5tr0khwize.webp)

CUDA API 更像是C 语言的扩展，以便最小化学习的时间。CUDA 提供一般DRAM 内存寻址方式：“发散” 和“聚集”内存操作，如图所示。从而提供最大的编程灵活性。从编程的观点来看，它可以在DRAM的任何区域进行读写数据的操作，就像在CPU 上一样。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.5q7ems4yo1.webp)

CUDA 允许并行数据缓冲或者在On-chip 内存共享，可以进行快速的常规读写存取，在线程之间共享数据。如图所示，应用程序可以最小化数据到DRAM 的overfetch 和round-trips ，从而减少对DRAM 内存带宽的依赖。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.4g4hggojzv.webp)

## cuda 编程模型

### 介绍

当通过CUDA 编译时，GPU 可以被视为能执行非常高数量并行线程的计算设备。它作为主CPU 的一个协处理器。换句话说，运行在主机上的并行数据和高密度计算应用程序部分，被卸载到这个设备上。

更准确地讲，一个被执行许多次不同数据的应用程序部分，可以被分离成为一个有很多不同线程在设备上执行的函数。达到这个效果，这个函数被编译成设备的指令集（kernel 程序），被下载到设备上。主机和设备使用它们自己的DRAM，主机内存和设备内存。并可以通过利用设备高性能直接内存存取(DMA)的引擎（API）从一个DRAM 复制数据到其他DRAM。

### 线程模型

主机发送一个连续的kernel 调用到设备。每个kernel 作为一个由线程块组成的批处理线程来执行。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.1lbtaodun4.webp)

一个线程块是一个线程的批处理，它通过一些快速的共享内存有效地分享数据并且在制定的内存访问中同步它们的执行。更准确地说，它可以在Kernel 中指定同步点，一个块里的线程被挂起直到它们所有都到达同步点。

一个`block` 可以包含的线程最大数量是有限的。然而，执行同一个kernel 的`block`可以合成一批线程块的`grid`，因此通过单一kernel 发送的请求的线程总数可以是非常巨大的。线程协作的减少会造成性能的损失，因为来自同一个`grid`的不同线程块中的线程彼此之不间能通讯和同步。这个模式允许kernel 用不同的并行能力有效地运行在各种设备上而不用再编译：一个设备可以序列地运行`grid`的所有块，如果它有非常少的并行特性，或者并行地运行，如果它有很多的并行的特性，或者通常是二者的组合。

### 内存模型

一条执行在设备上的线程，只允许通过如下的内存空间使用设备的DRAM 和On-Chip 内存：

- 读写每条线程的寄存器

- 读写每条线程的本地内存

- 读写每个`block`的共享内存

- 读写每个`grid`的全局内存

- 读写每个`grid`的常量内存

- 读写每个 `grid` 的纹理内存

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.83a13zz6vp.webp)

全局，常量，和纹理内存空间可以通过主机或者同一应用程序持续的通过kernel 调用来完成读取或写入。

全局，常量，和纹理内存空间对不同内存的用法加以优化。纹理内存同样提供不同的寻址模式，也为一些特殊的数据格式进行数据过滤。


## 硬件实现

### 一组带有 on-chip 共享内存的 SIMD 多处理器

设备可以被看作一组多处理器，如图所示。每个多处理器使用单一指令，多数据架构(SIMD) ：在任何给定的时钟周期内，多处理器的每个处理器执行同一指令，但操作不同的数据。

每个多处理器使用四个以下类型的on-chip 内存：

- 每个处理器一组本地32位寄存器

- 并行数据缓存或共享内存，被所有处理器共享实现内存空间共享，

- 通过设备内存的一个只读区域，一个只读常量缓冲器被所有处理器共享，

- 通过设备内存的一个只读区域，一个只读纹理缓冲器被所有处理器共享，

本地和全局内存空间作为设备内存的读写区域，而不被缓冲。每个多处理器通过纹理单元访问纹理缓冲器，它执行各种各样的寻址模式和数据过滤。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.4n7pbwv0my.webp)


### 执行模式

一个线程块`grid`是通过多处理器规划执行的。每个多处理器一个接一个的处理块批处理。一个块只被一个多处理器处理，因此可以对驻留在on-chip 共享内存中的共享内存空间形成非常快速的访问。一个批处理中每一个多处理器可以处理多少个块，取决于每个线程中分配了多少个寄存器和已知内核中每个时钟需要多少的共享内存，因为多处理器的寄存器和内存在所有的线程中是分开的。如果在至少一个块中，每个多处理器没有足够的寄存器或共享内存可用，那么内核将无法启动。

线程块在一个批处理中被一个多处理器执行，被称作`active`。每个`active` 块被划分成为SIMD 线程组，称为`warps`; 每一条这样的 `warp` 包含数量相同的线程，叫做 `warp` 大小，并且在SIMD 方式下通过多处理器执行; 线程调度程序周期性地从一条 `warp` 切换到另一条 `warp`，以达到多处理器计算资源使用的最大化。

块被划分成为`warp` 的方式总是相同的; 每条`warp` 包含连续的线程，线程索引从第一个 `warp` 包含着的线程0 开始递增。

一个多处理器可以处理并发地几个块，通过划分在它们之中的寄存器和共享内存。更准确地说，每条线程可使用的寄存器数量，等于每个多处理器寄存器总数除以并发的线程数量，并发线程的数量等于并发块的数量乘以每块线程的数量。

在一个块内的`warp` 次序是未定义的，但通过协调全局或者共享内存的存取，它们可以同步的执行。如果一个通过`warp` 线程执行的指令写入全局或共享内存的同一位置，写的次序是未定义的。

在一个线程块`grid`内的块次序是未定义的，并且在块之间不存在同步机制，因此来自同一个`grid`的二个不同块的线程不能通过全局内存彼此安全地通讯。
