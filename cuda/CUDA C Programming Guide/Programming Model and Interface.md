---
date: 2024-10-16 20:35:39
date modified: 2024-10-17 11:03:43
title: Programming Model and Interface
tags: 
categories:
---
# Programming Model
## Kernels

CUDA C++通过允许程序员定义称为内核的C++函数来扩展C++，这些函数在被调用时，由N个不同的CUDA线程并行执行N次，这与只执行一次的常规C++函数不同。

内核是通过使用`__global__`声明说明符定义的，并且给定内核调用的执行该内核的CUDA线程数是通过使用新的`<<<...>>>`执行配置语法指定的（参见C++语言扩展）。执行内核的每个线程都被赋予一个唯一的线程ID，这个ID可以通过内置变量在内核内部访问。

```c++
// Kernel definition

__global__ void VecAdd(float* A, float* B, float* C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main() {
	...
	// Kernel invocation with N threads
	VecAdd<<<1, N>>>(A, B, C);
	...
}
```

## Thread Hierarchy

为了方便起见，threadIdx是一个三维向量，因此可以使用一维、二维或三维的线程索引来识别线程，形成一维、二维或三维的线程块，称为线程块。这提供了一种自然的方式来在诸如向量、矩阵或体积等域的元素上调用计算。

```c++
/ Kernel definition

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

	int main() {
	...
	// Kernel invocation with one block of N * N * 1 threads
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```

  
线程块中的线程数量是有限制的，因为一个块中的所有线程都预期驻留在同一个流处理器核心上，并且必须共享该核心有限的内存资源。在当前的GPU上，一个线程块可以包含最多1024个线程。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.99tcd9b9ql.webp)

然而，一个 `kernel` 可以被多个形状相同的线程块执行，因此总线程数等于每个块的线程数乘以块的数量。`grid`中的每个块可以通过一维、二维或三维的唯一索引来识别，该索引在内核中可以通过内置的blockIdx变量访问。线程块的维度在内核中可以通过内置的blockDim变量访问。 将先前的MatAdd()示例扩展到处理多个块，代码如下所示。

```c++
// Kernel definition

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N)
		C[i][j] = A[i][j] + B[i][j];
}

int main() {
	...
	// Kernel invocation
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
	...
}
```


线程块需要独立执行：它们必须能够以任何顺序执行，无论是并行还是串行。这种独立性的要求允许线程块在任何数量的核心上以任何顺序调度。

块内的线程可以通过共享内存共享数据，并通过同步它们的执行来协调内存访问。更准确地说，可以在内核中通过调用__syncthreads()内置函数指定同步点；`__syncthreads()` 充当一个屏障，块中的所有线程都必须在此处等待，然后才能继续执行。共享内存给出了使用共享内存的一个例子。除了`__syncthreads()`之外，协作组API还提供了一套丰富的线程同步原语。

为了高效协作，共享内存预计是位于每个处理器核心附近的低延迟内存（类似于L1缓存），并且`__syncthreads()`预计是轻量级的。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.3uutuu2kn0.webp)

随着NVIDIA计算能力9.0的引入，CUDA编程模型引入了一个可选的层次结构级别，称为线程块集群，由线程块组成。类似于线程块中的线程被保证在一个流处理器上共同调度，集群中的线程块也被保证在GPU的处理簇（GPU Processing Cluster，简称GPC）上共同调度。

类似于线程块，集群也组织成一维、二维或三维，如线程块集群`grid`所示。集群中的线程块数量可以由用户定义，CUDA支持的最大集群大小为8个线程块，作为可移植的集群大小。请注意，在无法支持8个多处理器的GPU硬件或MIG配置上，最大集群大小将相应减少。识别这些较小配置以及支持超过8个线程块集群大小的大型配置是特定于架构的，可以使用`cudaOccupancyMaxPotentialClusterSize` API进行查询。

线程块集群可以在内核中使用编译时内核属性 `__cluster_dims__(X,Y,Z)` 启用，或者使用 CUDA 内核启动 API `cudaLaunchKernelEx`。下面的例子展示了如何使用编译时内核属性启动一个集群。使用内核属性设置的集群大小在编译时是固定的，然后可以使用传统的 `<<< >>>` 启动内核。如果一个内核使用编译时的集群大小，那么在启动内核时不能修改集群大小。

```c++
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output) {

}

int main() {
	float *input, *output;
	// Kernel invocation with compile time cluster size
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
	// The grid dimension is not affected by cluster launch, and is still enumerated
	// using number of blocks.
	// The grid dimension must be a multiple of cluster size.
	cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

  
线程块集群大小也可以在运行时设置，并且可以使用CUDA内核启动API `cudaLaunchKernelEx`来启动内核。下面的代码示例展示了如何使用可扩展API来启动集群内核。

```c++
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output) {

}

int main() {
	float *input, *output;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);

	// Kernel invocation with runtime cluster size
	{
	cudaLaunchConfig_t config = {0};
	// The grid dimension is not affected by cluster launch, and is still enumerated
	// using number of blocks.
	// The grid dimension should be a multiple of cluster size.
	config.gridDim = numBlocks;
	config.blockDim = threadsPerBlock;
	
	cudaLaunchAttribute attribute[1];
	attribute[0].id = cudaLaunchAttributeClusterDimension;
	attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
	attribute[0].val.clusterDim.y = 1;
	attribute[0].val.clusterDim.z = 1;
	config.attrs = attribute;
	config.numAttrs = 1;
	
	cudaLaunchKernelEx(&config, cluster_kernel, input, output);
	}
}
```

在具备计算能力9.0的GPU上，集群中的所有线程块都保证在单个GPU处理簇（GPC）上共同调度，这使得线程块之间可以进行高效的协作。这些线程块能够使用集群组API `cluster.sync()` 执行硬件支持的同步操作。集群组API还提供了以下成员函数来查询集群组的大小：

- `num_threads()`：这个API返回集群组中的总线程数。
- `num_blocks()`：这个API返回集群组中的总线程块数。

此外，可以查询集群组中线程或块的排名，使用以下API：

- `dim_threads()`：这个API用于查询当前线程在集群组中的排名。
- `dim_blocks()`：这个API用于查询当前线程块在集群组中的排名。

属于一个集群的线程块有权访问分布式共享内存（Distributed Shared Memory）。在集群中的线程块具有以下能力：

- 读取分布式共享内存中的任何地址。
- 写入分布式共享内存中的任何地址。
- 在分布式共享内存中的任何地址上执行原子操作。

分布式共享内存的一个示例应用是执行直方图统计。在这种应用中，集群中的每个线程块可以负责直方图的一部分，并将结果写入分布式共享内存。由于线程块之间可以高效地同步和共享数据，因此可以快速地完成整个直方图的计算。这种内存模型特别适合于需要大量线程间协作的计算任务，如数据并行处理、图像处理和某些类型的机器学习算法。

## Memory Hierarchy

CUDA线程在执行过程中可以从多个内存空间访问数据，如图所示。每个线程都有私有的本地内存。每个线程块都有共享内存，该共享内存对于块内的所有线程可见，并且与块具有相同的生命周期。线程块集群中的线程块可以对彼此的共享内存执行读、写和原子操作。所有线程都可以访问相同的全局内存。

还有两个额外的只读内存空间可供所有线程访问：常量和纹理内存空间。全局、常量和纹理内存空间针对不同的内存使用进行了优化（请参见设备内存访问）。纹理内存还为某些特定数据格式提供了不同的寻址模式，以及数据过滤（请参见纹理和表面内存）。全局、常量和纹理内存空间在相同应用程序启动内核期间是持久的。

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.60u8gmifgj.webp)

## Heterogeneous Programming

CUDA编程模型假设CUDA线程在一个物理上独立的设备上执行，该设备作为运行C++程序的主机的协处理器。例如，当内核在GPU上执行，而C++程序的其他部分在CPU上执行时，就是这种情况。

CUDA编程模型还假设主机和设备各自在DRAM中维护自己的独立内存空间，分别称为主机内存和设备内存。因此，程序通过调用CUDA运行时（在编程接口中描述）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存的分配和释放以及主机与设备内存之间的数据传输。

统一内存提供了托管内存，以桥接主机和设备内存空间。托管内存可以从系统中的所有CPU和GPU作为单个、连贯的内存镜像访问，具有共同的地址空间。这种能力使得设备内存超量分配成为可能，并且可以通过消除在主机和设备上显式镜像数据的需求，极大地简化应用程序移植的任务。有关统一内存的介绍，请参见统一内存编程。

## Asynchronous SIMT Programming Model

在CUDA编程模型中，线程是进行计算或内存操作的最底层抽象。从基于NVIDIA安培GPU架构的设备开始，CUDA编程模型通过异步编程模型为内存操作提供加速。异步编程模型定义了异步操作相对于CUDA线程的行为。

异步编程模型定义了异步屏障（Asynchronous Barrier）的行为，用于CUDA线程之间的同步。该模型还解释并定义了如何使用cuda::memcpy_async在GPU计算的同时从全局内存异步移动数据。

### Asynchronous Operations 

异步操作是指由CUDA线程发起的操作，该操作似乎是由另一个线程异步执行。在结构良好的程序中，一个或多个CUDA线程将与异步操作同步。发起异步操作的CUDA线程不一定需要是同步线程之一。 这样的异步线程（假设的线程）总是与发起异步操作的CUDA线程相关联。异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理（例如，`cuda::memcpy_async`），也可以在库内部隐式管理（例如，`cooperative_groups::memcpy_async`）。 同步对象可以是`cuda::barrier`或`cuda::pipeline`。这些对象在异步屏障和`cuda::pipeline`的异步数据复制中有详细说明。这些同步对象可以在不同的线程作用域中使用。作用域定义了可以使用同步对象与异步操作同步的线程集合。以下表格定义了CUDA C++中可用的线程作用域以及可以与每个作用域同步的线程：

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/cuda/image.2a52ve39sq.webp)




# Programming Interface
  
CUDA C++为熟悉C++编程语言的用户提供了一条简单的路径，可以轻松地为设备编写程序。 它包括对C++语言的最小集合扩展和一个运行时库。

## Compilation with NVCC

内核可以使用称为PTX的CUDA指令集架构来编写，这在PTX参考手册中有描述。然而，通常使用像C++这样的高级编程语言会更有效。在这两种情况下，内核必须通过nvcc编译成二进制代码才能在设备上执行。

nvcc是一个编译器驱动程序，它简化了编译C++或PTX代码的过程：它提供了简单且熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。本节概述了nvcc的工作流程和命令选项。完整的描述可以在nvcc用户手册中找到。

### Compilation Workflow

#### Offline Compilation

使用nvcc编译的源文件可以包含主机代码（即，在主机上执行的代码）和设备代码（即，在设备上执行的代码）的混合。nvcc的基本工作流程包括从主机代码中分离设备代码，然后执行以下步骤：

- 将设备代码编译成汇编形式（PTX代码）和/或二进制形式（cubin对象）， 

-  通过替换在内核中引入的<<<…>>>语法（在执行配置中更详细描述）来修改主机代码，替换为必要的CUDA运行时函数调用，以从PTX代码和/或cubin对象加载和启动每个编译后的内核。

修改后的主机代码可以输出为C++代码，留待使用其他工具编译，或者直接输出为对象代码，通过在最后编译阶段让nvcc调用主机编译器。

应用程序然后可以执行以下操作：

- 或者链接到编译后的主机代码（这是最常见的情况），

- 或者忽略修改后的主机代码（如果有的话）并使用CUDA驱动API（参见驱动API）来加载和执行PTX代码或cubin对象。

#### Just-in-Time Compilation

应用程序在运行时加载的任何PTX代码都会由设备驱动程序进一步编译成二进制代码，这个过程称为即时编译。即时编译会增加应用程序的加载时间，但允许应用程序利用每次新设备驱动程序带来的编译器改进。这也是应用程序在编译时还不存在的设备上运行的唯一方式，详细内容请参见应用程序兼容性。

当设备驱动程序为某个应用程序即时编译一些PTX代码时，它会自动缓存生成的二进制代码的副本，以避免在应用程序的后续调用中重复编译。这个缓存被称为计算缓存，当设备驱动程序升级时，计算缓存会自动失效，以便应用程序可以从设备驱动程序内置的新即时编译器的改进中受益。

环境变量可用于控制即时编译，具体描述请参见CUDA环境变量。

作为使用nvcc编译CUDA C++设备代码的替代方案，NVRTC可以在运行时用于将CUDA C++设备代码编译成PTX。NVRTC是一个用于CUDA C++的运行时编译库；更多信息可以在NVRTC用户指南中找到。

### Binary Compatibility

二进制代码是特定于架构的。cubin对象是通过使用编译器选项`-code`来生成的，该选项指定了目标架构：例如，使用`-code=sm_80`进行编译会为计算能力为8.0的设备生成二进制代码。从一个小版本到下一个版本的二进制兼容性是有保证的，但是从一个小版本到前一个版本或者跨主要版本则没有保证。换句话说，为计算能力X.y生成的cubin对象只能在计算能力为X.z的设备上执行，其中z ≥ y。

### PTX Compatibility

一些PTX指令只在计算能力较高的设备上支持。例如，Warp Shuffle函数仅在计算能力为5.0及以上的设备上支持。编译器选项`-arch`用于指定在将C++代码编译为PTX代码时假设的计算能力。因此，例如包含warp shuffle的代码必须使用`-arch=compute_50`（或更高）进行编译。为特定计算能力生成的PTX代码总是可以编译为更大或等于该计算能力的二进制代码。

请注意，从早期PTX版本编译的二进制文件可能不会使用某些硬件特性。例如，针对计算能力为7.0（Volta）的设备编译的二进制文件，如果是从为计算能力6.0（Pascal）生成的PTX编译的，将不会使用Tensor Core指令，因为这些在Pascal上不可用。因此，最终的二进制文件可能性能不如使用最新版本PTX生成的二进制文件。

编译为目标架构条件特征的PTX代码只在完全相同的物理架构上运行，而在其他地方则不会运行。架构条件PTX代码不具备向前和向后兼容性。例如，使用sm_90a或compute_90a编译的代码只在计算能力为9.0的设备上运行，并且不具备向后或向前兼容性。


## CUDA Runtime

运行时是在cudart库中实现的，该库与应用程序链接，可以是静态链接通过cudart.lib或libcudart.a，也可以是动态链接通过cudart.dll或libcudart.so。需要cudart.dll和/或libcudart.so进行动态链接的应用程序通常将它们作为应用程序安装包的一部分包含。只有当组件链接到同一实例的CUDA运行时时，传递CUDA运行时符号的地址才是安全的。它的所有入口点都以前缀cuda开头。

### Initialization

从CUDA 12.0开始，`cudaInitDevice()`和`cudaSetDevice()`调用用于初始化运行时和与指定设备关联的主上下文。如果没有这些调用，运行时会隐式使用设备0，并根据需要自行初始化以处理其他运行时API请求。在计时运行时函数调用和解释第一次调用运行时返回的错误代码时，需要记住这一点。在12.0之前，`cudaSetDevice()`不会初始化运行时，应用程序通常会使用无操作的运行时调用`cudaFree(0)`来将运行时初始化与其他API活动隔离开（这样做既是为了计时，也是为了错误处理）。

运行时为系统中的每个设备创建一个CUDA上下文。这个上下文是该设备的主上下文，在第一次需要在此设备上激活上下文的运行时函数时被初始化。它被应用程序的所有主机线程共享。作为上下文创建的一部分，如果需要，设备代码会被即时编译并加载到设备内存中。这一切都是透明发生的。如果需要，例如，为了与驱动API互操作，可以通过驱动API访问设备的主上下文。

当主机线程调用`cudaDeviceReset()`时，这会销毁主机线程当前操作的设备的主上下文（即，如设备选择中定义的当前设备）。任何将此设备作为当前设备的主机线程进行的下一次运行时函数调用都会为该设备创建一个新的主上下文。

### Device Memory

CUDA编程模型假设系统由主机和设备组成，每个都有自己的独立内存。内核在设备内存中操作，因此运行时提供了用于分配、释放和复制设备内存的函数，以及用于在主机内存和设备内存之间传输数据的函数。

设备内存可以分配为线性内存或CUDA数组。

CUDA数组是为纹理提取优化的不透明内存布局。

线性内存是在单个统一地址空间中分配的，这意味着单独分配的实体可以通过指针相互引用，例如，在二叉树或链表中。地址空间的大小取决于主机系统（CPU）和使用GPU的计算能力：

线性内存分配是通过`cudaMalloc()`函数进行的，而释放是通过`cudaFree()`函数。数据在主机内存和设备内存之间的传输是通过`cudaMemcpy()`函数系列来完成的，这些函数支持各种传输模式，包括主机到设备、设备到主机以及设备到设备的传输。

在CUDA中，线性内存通常使用`cudaMalloc()`来分配，使用`cudaFree()`来释放。主机内存与设备内存之间的数据传输通常使用`cudaMemcpy()`来完成。在《Kernels》部分的向量加法代码示例中，需要将向量从主机内存复制到设备内存。

```c++
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	C[i] = A[i] + B[i];
}

// Host code
int main() {
	int N = ...;
	size_t size = N * sizeof(float);
	
	// Allocate input vectors h_A and h_B in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
	
	// Initialize input vectors
	...
	
	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) ∕ threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// Free host memory
	...
}
```
`   cudaMallocPitch()` 和 `cudaMalloc3D()` 函数用于分配二维或三维数组，它们确保分配的内存适当地填充以满足《设备内存访问》中描述的对齐要求，从而在访问行地址或使用 `cudaMemcpy2D()` 和 `cudaMemcpy3D()` 函数在二维数组与其他设备内存区域之间执行复制时确保最佳性能。返回的跨度（或步幅）必须用于访问数组元素。
```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height) {
	for (int r = 0; r < height; ++r) {
		float* row = (float*)((char*)devPtr + r * pitch);
		for (int c = 0; c < width; ++c) {
			float element = row[c];
		}
	}
}
```
  
以下代码示例分配了一个宽度 x 高度 x 深度的浮点值三维数组，并展示了如何在设备代码中遍历数组元素：
```c++
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth) {
	char* devPtr = devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch * height;
	for (int z = 0; z < depth; ++z) {
		char* slice = devPtr + z * slicePitch;
		for (int y = 0; y < height; ++y) {
			float* row = (float*)(slice + y * pitch);
			for (int x = 0; x < width; ++x) {
				float element = row[x];
			}
		}
	}
}
```
  
为了避免分配过多内存从而影响系统范围内的性能，可以根据问题大小从用户那里请求分配参数。如果分配失败，您可以回退到其他较慢的内存类型（`cudaMallocHost()`、`cudaHostRegister() `等），或者返回一个错误，告诉用户需要多少内存被拒绝了。如果您的应用程序由于某种原因无法请求分配参数，我们建议对于支持的平台使用 `cudaMallocManaged()`。

参考手册列出了所有用于在以下内存之间复制数据的各种函数：使用 `cudaMalloc() `分配的线性内存、使用 `cudaMallocPitch() `或` cudaMalloc3D() `分配的线性内存、CUDA 数组，以及为在全局或常量内存空间中声明的变量分配的内存。以下代码示例展示了通过运行时 API 访问全局变量的不同方式：

```c++
__constant__ float constData[256];
float data[256];

cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```
`cudaGetSymbolAddress()` 用于检索指向在全局内存空间中声明的变量所分配内存的地址。通过 `cudaGetSymbolSize() `获取已分配内存的大小。

### Device Memory L2 Access Management

当CUDA内核重复访问全局内存中的数据区域时，这样的数据访问可以被认为是持久的。另一方面，如果数据只被访问一次，这样的数据访问可以被认为是流式的。

从CUDA 11.0开始，计算能力为8.0及以上的设备具有影响L2缓存中数据持久性的能力，这可能会提高访问全局内存的带宽并降低延迟。

#### L2 Cache Set-Aside for Persisting Accesses

可以将L2缓存的一部分设置保留，用于持久访问全局内存。持久访问将优先使用这部分保留的L2缓存，而普通或流式访问全局内存只能在持久访问未使用这部分L2缓存时使用它。持久访问的L2缓存保留大小可以在一定范围内进行调整。

```c++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3∕4 of L2 cachefor persisting accesses or the max allowed */
```

当GPU配置为多实例GPU（MIG）模式时，L2缓存保留功能将被禁用。在使用多进程服务（MPS）时，不能通过cudaDeviceSetLimit更改L2缓存的保留大小。相反，保留大小只能在启动MPS服务器时通过环境变量CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT指定。

#### L2 Policy for Persisting Accesses

访问策略窗口指定全局内存中的一个连续区域以及该区域内访问的L2缓存中的持久性属性。以下代码示例展示了如何使用CUDA流设置一个L2持久访问窗口：

```c++
cudaStreamAttrValue stream_attribute;
//Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); 
//Global Memory data pointer 
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
// Number of bytes for persistence access.

// (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio = 0.6;
// Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp= cudaAccessPropertyPersisting; 
// Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; 
// Type of access property on cache miss.

// Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

当随后在CUDA流中执行内核时，位于全局内存范围 `[ptr…ptr+num_bytes) `内的内存访问比访问其他全局内存位置更有可能保留在L2缓存中。

L2持久性也可以为CUDA图内核节点设置，如下例所示：

```c++
cudaKernelNodeAttrValue node_attribute;
// Kernellevel attributes data structure
node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); 
// Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;
// Number of bytes for persistence access.

// (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio = 0.6;
// Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp= cudaAccessPropertyPersisting; 
// Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; 
// Type of access property on cache miss.
// Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

命中率参数（hitRatio）可用于指定接收到hitProp属性的访问比例。在上述两个示例中，全局内存区域`[ptr…ptr+num_bytes)`内60%的内存访问具有持久化属性，而40%的内存访问具有流式属性。具体哪些内存访问被归类为持久化（即hitProp）是随机的，概率大约为hitRatio；概率分布取决于硬件架构和内存范围。

例如，如果L2预留缓存大小为16KB，且accessPolicyWindow中的num_bytes为32KB：

- 当hitRatio为0.5时，硬件将随机选择32KB窗口中的16KB作为持久化数据，并将其缓存在预留的L2缓存区域中。
- 当hitRatio为1.0时，硬件将尝试将整个32KB窗口缓存在预留的L2缓存区域中。由于预留区域小于窗口大小，将会驱逐缓存行以保持最近使用的16KB数据在L2缓存的预留部分中。

因此，hitRatio可用于避免缓存行的频繁替换，并总体减少进出L2缓存的数据量。

hitRatio值低于1.0可以用来手动控制不同CUDA流的`accessPolicyWindow`s可以在L2中缓存的数据量。例如，假设L2预留缓存大小为16KB；两个不同CUDA流中的并发内核，每个内核都有一个16KB的`accessPolicyWindow`，且hitRatio值都为1.0，它们可能会在竞争共享L2资源时驱逐对方的缓存行。然而，如果两个`accessPolicyWindows`的hitRatio值都为0.5，它们将不太可能驱逐自己的或对方的持久化缓存行。

#### L2 Access Properties

定义了三种不同全局内存数据访问的访问属性：

1. `cudaAccessPropertyStreaming`：具有流式属性的内存访问不太可能保留在L2缓存中，因为这些访问优先被驱逐。

2. `cudaAccessPropertyPersisting`：具有持久化属性的内存访问更有可能保留在L2缓存中，因为这些访问优先保留在L2缓存的预留部分。

3. `cudaAccessPropertyNormal`：这个访问属性强制将先前应用的持久化访问属性重置为正常状态。来自先前CUDA内核的具有持久化属性的内存访问可能会在它们预期使用后很长时间内保留在L2缓存中。这种使用后持久性减少了L2缓存中可供后续不使用持久化属性的内核使用的缓存量。使用`cudaAccessPropertyNormal`属性重置访问属性窗口，移除了先前访问的持久化（优先保留）状态，就像先前的访问没有访问属性一样。

#### L2 Persistence Example

以下示例展示了如何为持久化访问预留L2缓存，通过CUDA流在CUDA内核中使用预留的L2缓存，然后重置L2缓存。

```c++
cudaStream_t stream;
cudaStreamCreate(&stream); 
// Create CUDA stream
cudaDeviceProp prop; 
// CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id); 
// Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 
// set-aside 3∕4 of L2 cache for persisting accesses or the max allowed
size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes); 
// Select minimum of user defined num_bytes and max window size.
cudaStreamAttrValue stream_attribute; 
// Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(data1); 
// Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size; 
// Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio = 0.6; 
// Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; 
// Persistence Property
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; 
// Type of access property on cache miss
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
// Set the attributes to a CUDA Stream
for(int i = 0; i < 10; i++) {
	cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1); 
	// This data1 is used by a kernel multiple times
}
// [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1); 
// A different kernel in the same stream can also benefit
// from the persistence of data1
stream_attribute.accessPolicyWindow.num_bytes = 0;
// Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
// Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();
// Remove any persistent lines in L2
cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);
// data2 can now benefit from full L2 in normal mode
```

#### Reset L2 Access to Normal

持久化L2缓存行可能在上一个CUDA内核使用后很长时间内仍然保留在L2缓存中。因此，将L2缓存重置为正常状态对于流式或普通内存访问来说很重要，以便它们能够以正常优先级使用L2缓存。有三种方法可以将持久化访问重置为正常状态。

1. 使用访问属性`cudaAccessPropertyNormal`重置先前的持久化内存区域。

2. 通过调用`cudaCtxResetPersistingL2Cache()`将所有持久化L2缓存行重置为正常状态。

3. 最终未被触碰的缓存行将自动重置为正常状态。强烈建议不要依赖自动重置，因为自动重置发生所需的时间是不确定的。

#### Manage Utilization of L2 set-aside cache

在不同的CUDA流中并发执行的多个CUDA内核可能为其流分配了不同的访问策略窗口。然而，预留的L2缓存部分是在所有这些并发CUDA内核之间共享的。因此，这个预留缓存部分的净利用率是所有并发内核个体使用的总和。随着持久化访问量的增加，超出预留L2缓存容量的部分，将设计内存访问为持久化的好处会减少。

为了管理预留L2缓存部分的使用，应用程序必须考虑以下因素： 

- 预留L2缓存的大小。

- 可能并发执行的CUDA内核。

- 可能并发执行的所有CUDA内核的访问策略窗口。

- 何时以及如何需要重置L2缓存，以允许正常或流式访问以平等优先级使用之前预留的L2缓存。

#### Query L2 cache Properties

与L2缓存相关的属性是cudaDeviceProp结构体的一部分，可以使用CUDA运行时API cudaGetDeviceProperties进行查询。CUDA设备属性包括以下内容：

- `l2CacheSize`: GPU上可用的L2缓存总量。

- `persistingL2CacheMaxSize`: 可以为持久化内存访问预留的最大L2缓存量。

- `accessPolicyMaxWindowSize`: 访问策略窗口的最大大小。

#### Control L2 Cache Set-Aside Size for Persisting Memory Access

为持久化内存访问预留的L2缓存大小可以通过CUDA运行时API `cudaDeviceGetLimit` 查询，并使用 `cudaDeviceSetLimit` API作为 `cudaLimit` 设置。设置此限制的最大值是 `cudaDeviceProp::persistingL2CacheMaxSize`。

```c++
enum cudaLimit {
/* other fields not shown */
cudaLimitPersistingL2CacheSize
};
```

### Shared Memory

如变量内存空间说明符中详细描述的，共享内存是使用 `__shared__` 内存空间说明符来分配的。

共享内存预计比全局内存快得多，这在线程层次结构中提到，并在共享内存部分详细说明。它可以被用作暂存内存（或软件管理的缓存），以减少CUDA块从全局内存的访问，如下面的矩阵乘法示例所示。

以下代码样本是一个不利用共享内存的矩阵乘法的直接实现。每个线程读取矩阵A的一行和矩阵B的一列，并计算矩阵C对应的元素，如图8所示。因此，矩阵A从全局内存中读取B.width次，矩阵B从全局内存中读取A.height次。

以下是矩阵乘法的示例代码，该代码没有使用共享内存：

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height; 
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);
	
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width ∕ dimBlock.x, A.height ∕ dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}
```

以下代码示例是利用共享内存实现的矩阵乘法。在这个实现中，每个线程块负责计算矩阵C的一个方形子矩阵Csub，而块内的每个线程负责计算Csub的一个元素。如图9所示，Csub等于两个矩形矩阵的乘积：一个是维度为(A.width, block_size)的A的子矩阵，其行索引与Csub相同；另一个是维度为(block_size, A.width)的B的子矩阵，其列索引与Csub相同。为了适应设备的资源，这两个矩形矩阵被划分为尽可能多的维度为block_size的方形矩阵，并计算这些方形矩阵乘积的和来得到Csub。每个这些乘积都是由一个线程从全局内存加载两个对应的方形矩阵到共享内存（每个矩阵一个元素），然后每个线程计算乘积的一个元素。每个线程将这些乘积的结果累加到寄存器中，完成后将结果写入全局内存。

通过这种方式阻塞计算，我们利用了快速的共享内存，并且由于A只从全局内存中读取(B.width / block_size)次，B只读取(A.height / block_size)次，从而节省了大量全局内存带宽。

以下是对先前代码样本中的矩阵类型进行扩充，添加了步长字段，以便可以高效地用相同的类型表示子矩阵。使用`__device__`函数来获取和设置元素，以及从矩阵构建任何子矩阵。

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);
	
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width ∕ dimBlock.x, A.height ∕ dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
	cudaMemcpyDeviceToHost);
	
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	
	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	
	// Each thread computes one element of Csub by accumulating results into Cvalue
	float Cvalue = 0;
	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	// Loop over all the sub-matrices of A and B that are required to compute Csub
	// Multiply each pair of sub-matrices together and accumulate the results
	for (int m = 0; m < (A.width ∕ BLOCK_SIZE); ++m) {
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];
			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}
	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}
```

### Distributed Shared Memory

在计算能力9.0中引入的线程块集群（Thread Block Clusters）提供了线程块集群中的线程访问所有参与线程块的共享内存的能力。这种分区的共享内存被称为分布式共享内存（Distributed Shared Memory），相应的地址空间被称为分布式共享内存地址空间。属于线程块集群的线程可以在分布式地址空间中进行读取、写入或执行原子操作，无论地址是否属于本地线程块还是远程线程块。无论内核是否使用分布式共享内存，共享内存的大小规格（静态或动态）仍然是每个线程块的。分布式共享内存的大小只是每个集群中的线程块数量乘以每个线程块的共享内存大小。

访问分布式共享内存中的数据需要所有线程块都存在。用户可以使用集群组API中的cluster.sync()来确保所有线程块都已开始执行。用户还需要确保所有分布式共享内存操作在线程块退出之前完成，例如，如果远程线程块试图读取给定线程块的共享内存，用户需要确保远程线程块读取的共享内存完成后再退出。

CUDA提供了一种访问分布式共享内存的机制，应用程序可以利用其能力来获益。下面我们来看一个简单的直方图计算，以及如何使用线程块集群在GPU上对其进行优化。计算直方图的标准方法是在每个线程块的共享内存中进行计算，然后执行全局内存原子操作。这种方法的一个限制是共享内存的容量。一旦直方图箱子不再适合共享内存，用户需要直接在全局内存中计算直方图，因此需要进行原子操作。有了分布式共享内存，CUDA提供了一个中间步骤，根据直方图箱子的尺寸，直方图可以在共享内存、分布式共享内存或直接在全局内存中计算。

下面的CUDA内核示例展示了如何根据直方图箱子的数量在共享内存或分布式共享内存中计算直方图。

```c++
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size) {
	extern __shared__ int smem[];
	namespace cg = cooperative_groups;
	int tid = cg::this_grid().thread_rank();
	
	// Cluster initialization, size and calculating local bin offsets.
	cg::cluster_group cluster = cg::this_cluster();
	unsigned int clusterBlockRank = cluster.block_rank();
	int cluster_size = cluster.dim_blocks().x;
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		smem[i] = 0; //Initialize shared memory histogram to zeros
	}
	// cluster synchronization ensures that shared memory is initialized to zero in
	// all thread blocks in the cluster. It also ensures that all thread blocks
	// have started executing and they exist concurrently.
	cluster.sync();
	for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
		int ldata = input[i];
		// Find the right histogram bin.
		int binid = ldata;
		if (ldata < 0)
			binid = 0;
		else if (ldata >= nbins)
			binid = nbins - 1;
			
		//Find destination block rank and offset for computing
		//distributed shared memory histogram
		int dst_block_rank = (int)(binid ∕ bins_per_block);
		int dst_offset = binid % bins_per_block;
		
		//Pointer to target block shared memory
		int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
		
		//Perform atomic update of the histogram bin
		atomicAdd(dst_smem + dst_offset, 1);
		}
	// cluster synchronization is required to ensure all distributed shared
	// memory operations are completed and no thread block exits while
	// other thread blocks are still accessing distributed shared memory
	cluster.sync();
	// Perform global memory histogram, using the local distributed memory histogram
	int *lbins = bins + cluster.block_rank() * bins_per_block;
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		atomicAdd(&lbins[i], smem[i]);
	}
}
```

上述内核可以在运行时根据所需的分布式共享内存量来启动集群大小。如果直方图足够小，可以只适应一个块的共享内存，用户可以以集群大小1来启动内核。下面的代码片段展示了如何根据共享内存需求动态启动集群内核。

```c++
// Launch via extensible launch
{
	cudaLaunchConfig_t config = {0};
	config.gridDim = array_size ∕ threads_per_block;
	config.blockDim = threads_per_block;
	
	// cluster_size depends on the histogram size.
	// ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
	int cluster_size = 2; ∕∕ size 2 is an example here
	int nbins_per_block = nbins ∕ cluster_size;
	
	// dynamic shared memory size is per block.
	// Distributed shared memory size = cluster_size * nbins_per_block * sizeof(int)
	config.dynamicSmemBytes = nbins_per_block * sizeof(int);
	
	CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
	
	cudaLaunchAttribute attribute[1];
	attribute[0].id = cudaLaunchAttributeClusterDimension;
	attribute[0].val.clusterDim.x = cluster_size;
	attribute[0].val.clusterDim.y = 1;
	attribute[0].val.clusterDim.z = 1;
	
	config.numAttrs = 1;
	config.attrs = attribute;
	
	cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

### Page-Locked Host Memory

