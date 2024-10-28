---
date: 2024-10-19 19:32:12
date modified: 2024-10-20 16:44:09
title: Hardware Acceleration
tags:
  - course
categories:
  - DLSystem
date created: 2024-09-25 13:26:08
---
## General acceleration techniques

### Vectorization
```c++
void vecadd(float* A, float* B, float* C) {
	for (int i = 0; i < 64: i++) {
		float4 a = load_float4(A + i*4);
		float4 b = load_float4(B + i*4);
		float4 c = add_float4(a, b);
		store_float4(C + i*4, c);
	}
}
```

### Data layout and strides

Row major: `A[i,j]` $\Rightarrow$ `Adata[i * A.shape[1] + j]`

Column major: `A[i,j]` $\Rightarrow$ `Adata[j * A.shape[0] + i]`

Strides format: `A[i, j]` $\Rightarrow$ `Adata[i * A.strides[0] + j * strides[1]]`

**Advantages** : can perform transformation/slicing in zero copy way

- Slice: change the begin offset and shape

- Transpose: swap the strides

- Broadcast: insert a stride equals 0

**Disadvantages**: memory access becomes not continuous

- Makes vectorization harder

- Many linear algebra operations may require compact the array first.

### Parallelization
```c++
void vecadd(float* A, float* B, float* C) {
	#pragma omp parallel for
	for (int i = 0; i < 64; i++) {
		float4 a = load_float4(A + i*4);
		float4 b = load_float4(B + i*4);
		float4 c = add_float4(a, b);
		store_float4(C * 4, c);
	}
}
```

## Case study: matrix multiplication

### Vanilla matrix multiplication
Compute C = dot(A, B.T)

```c++
dram float A[n][n], B[n][n], c[n][n]
for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
		register float c = 0;
		for (int k = 0; k < n; k++) {
			register float a = A[i][k];
			register float b = B[j][k];
			c += dot(a, b.T);
		}
		C[i][j] = c;
	}
}
```

A's dram $\rightarrow$ register time cost: $n^3$
B's dram $\rightarrow$ register time cost: $n^3$
A's register memory cost: 1
B's register memory cost: 1
C's register memory cost: 1

**load cost**: $2 * \text{dramspeed} * n^3$
**Register cost**: 3

### Register tiled matrix multiplication
```c++
dram float A[n/v1][n/v3][v1][v3];
dram float B[n/v2][n/v3][v2][v3];
dram float C[n/v1][n/v2][v1][v2];

for (int i = 0;i < n/v1; i++) {
	for (int j = 0; j < n/v2; j++) {
		register float c[v1][v2]=0;
		for int (k = 0; k < n/v3; k++) {
			register float a[v1][v3] = A[i][k];
			register float b[v2][v3] = B[j][k];
			c += dot(a, b.T)
		}
		C[i][j] = c;
	}
}
```

A's dram $\rightarrow$ register time cost: $n^3/v_2$
B's dram $\rightarrow$ register time cost: $n^3/v_1$
A's register memory cost: $v_1*v_3$
B's register memory cost: $v_2*v_3$
C's register memory cost: $v_1*v_2$

**load cost**: $\text{dramspeed} * (n^3/v_2+n^3/v_1)$
**Register cost**: $v_1*v_3 + v_2*v_3+v_1*v_2$

### Cache line aware tiling
```c++
dram float A[n/b1][b1][n];
dram float B[n/b2][b2][n];
dram float C[n/b1][n/b2][b1][b2];
for (int i = 0; i < n; i++){
	l1cache float a[b1][n] = A[i];
	for (int j = 0; j < n/b2; j++) {
		l1cache b[b2][n] = B[j];
		
		C[i][j] = dot(a, b.T);
	}
}
```

A's dram $\rightarrow$ l1 time cost: $n^2$
B's dram $\rightarrow$ l1 time cost: $n^3/b_1$

**Constraints:**
$b_1*n+b_2*n<\text{l1 cache size}$
To still apply register blocking on dot
- $b_1\%v_1==0$
- $b_2\%v_2==0$

### Putting it together
```c++
dram float A[n/b1][b1/v1][n][v1];
dram float B[n/b2][b2/v2][n][v2];

for (int i = 0; i < n; i++) {
	l1cache float a[b1/v1][n][v1] = A[i];
	for (int j = 0; j < n/b2; j++) {
		l1cache b[b2/v2][n][v2]=B[j];
		for (int x = 0; x < b1/v1; x++) {
			for (int y = 0; y < b2/v2; y++) {
				register float c[v1][v2] = 0;
				for (int k = 0; k < n; k++) {
					register float ar[v1] = a[x][k][:];
					register float br[v2] = b[y][k][:];
					C += dot(ar, br.T)
				}
			}
		}
	}
}
```

**load cost**: $\text{l1speed}*(n^3/v2 + n^3/v1) + \text{dramspeed} * (n^2+n^3/b1)$

### Key insight: memory load reuse
```c++
dram float A[n/v1][n/v3][v1][v3];
dram float B[n/v2][n/v3][v2][v3];
dram float C[n/v1][n/v2][v1][v2];

for (int i = 0;i < n/v1; i++) {
	for (int j = 0; j < n/v2; j++) {
		register float c[v1][v2]=0;
		for int (k = 0; k < n/v3; k++) {
			register float a[v1][v3] = A[i][k];
			register float b[v2][v3] = B[j][k];
			c += dot(a, b.T)
		}
		C[i][j] = c;
	}
}
```

a get reused v2 times
b get reused v1 times

A's dram $\rightarrow$ register time cost: $n^3/v_2$
B's dram $\rightarrow$ register time cost: $n^3/v_1$

### Common reuse patterns
```c++
float A[n][n];
float B[n][n];
float B[n][n];

C[i][j] = sum(A[i][k] * B[j][k], axis=k)
```

Access of A is independent of j, 
tile the j dimension by v enables reuse of A for v times.

### Discuss: possible reuse pattern in convolution

```c++
float Input[n][ci][h][w];
float Weight[co][ci][K][K];
float Output[n][co][h][w];

Output[b][co][y][x] = 
	sum(Input[b][k][y+ry][x+rx]) * 
		Weight[co][k][ry][rx], axis=[k, ry, rx])
```

## GPU programming

### GPU programming mode: SIMT

- Single instruction multiple threads (SIMT)

- All threads executes the same code, but can take different path

- Threads are grouped into blocks, thread within the same block have shared memory

- Blocks are grouped into a launch grid

- A kernel executes a grid

```c++
void VecAddCPU(float* A, float* B, float* C, int n) {
	for (int i = 0; i < n; i++) {
		C[i] = A[i] + B[i];
	}
}

__global__ void VecAddKernel(float* A, float* B, float* C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}	
}

void VecAddCUDA(float* Acpu, float* Bcpu, float* Ccpu, int n) {
	float* dA, *dB, *dC;
	cudaMalloc(&dA, n * sizeof(float));
	cudaMalloc(&dB, n * sizeof(float));
	cudaMalloc(&dC, n * sizeof(float));
	cudaMemcpy(dA, Acpu, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, Bcpu, n * sizeof(float), cudaMemcpyHostToDevice);
	int threads_per_block = 512;
	int nblocks = (n + threads_per_block - 1) / threads_per_block;
	VecAddKernel<<<nblocks, thread_per_block>>>(dA, dB, dC, n);
	cudaMemcpy(Ccpu, dC, n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}
```

really application usually <mark>keep data in gpu memory as long as possible</mark>. 

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.8ad95j0hqg.webp)

Global memory: where data are stored when you call `cudaMalloc` and `cudaFree`.

### Example: window sum with shared memory

```c++
__global__ void WindowSumSharedKernel(float* A, float* B, int n) {
	__shared__ float temp[THREADS_PER_BLOCK + 2 * RADIUS];
	int base = blockDim.x * blockIdx.x;
	int out_idx = base + threadIdx.x;
	if (base + threadIdx.x < n) {
		temp[threadIdx.x] = A[base + threadIdx.x];
	}
	if (threadIdx.x < 2 * RADIUS && base + THREADS_PER_BLOCK + threadIdx.x < n) { 
		temp[threadIdx.x + THRE;ADS_PER_BLOCK] = A[base + THREADS_PER_BLOCK + threadIdx.x];
	}
	__syncthreads();
	if (out_idx < n) {
		float sum = 0;
		for (in dx = -RADIUS; dx <= RADIUS; dx++) {
			sum += temp[threadIdx.x + dx + RADIUS];
		}
		B[out_idx] = sum;
	}
}
```

<mark>Memory loading reuse</mark>, Launch thread and blocks, Cooperatively fetch common to shared memory to increase reuse.

## Case study: matrix multiplication on GPU

### Thread-level: register tiling

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
	int ybase = blockIdx.y * blockDim.y + threadIdx.y;
	int xbase = blockIdx.x * blockDim.x + threadIdx.x;
	
	float c[V][V] = {0};
	float a[V], b[V];
	for (int k = 0; k < N; k++) {
		a[:] = A[k, ybase * V : ybase * V + V];
		b[:] = B[k, xbase * V : xbase * V + V];
		for (int y = 0; y < V; y++) {
			for (int x = 0; x < V; x++) {
				c[y][x] += a[y] * b[x];
			}
		}
	}
	C[ybase * V : ybase * V + V, xbase * V : xbase * V + V] = c[:];
}
```
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.3goe9f5buv.webp)

### Block-level: shared memory tiling

```c++
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
	__shared__ float sA[S][L], sB[S][L];
	float c[V][V] = {0};
	float a[V], b[V];
	int yblock = blockIdx.y;
	int xblock = blockIdx.x;
	
	for (int ko = 0; ko < N; ko += S) {
		__syncthreads();
		// needs to be implemented by thread cooperatiev fetching
		sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
		sB[:, :] = B[k : k + S, xblock * L : xblock * L + L];
		__syncthreads();
		for (int ki = 0; ki < S; ki++) {
			a[:] = sA[ki, threadIdx.y * V : threadIdx.y * V + V];
			b[:] = sA[ki, threadIdx.x * V : threadIdx.x * V + V];
			for (int y = 0; y < V; y++) {
				for (int x = 0; x < V; x++) {
					c[y][x] += a[y] * b[x];
				}
			}
		}
	}
	int ybase = blockIdx.y * blockDim.y + threadIdx.y;
	int xbase = blockIdx.x * blockDim.x + threadIdx.x;
	C[ybase * V : ybase * V + V, xbase * V : xbase * V + V] = c[:];
}
```

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.4qrbfquydy.webp)

global $\rightarrow$ shared copy: $2 * N^3 / L$

shared $\rightarrow$ register: $2 * N^3 / V$

**Auto tuning**

### Expand Cooperative Fetching

```c++
sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
```

become

```c++
int nthreads = blockDim.y * blockDim.x;
int tid = threadIDx.y * blockDim.x + threadIdx.x;

for (int j = 0; j < L * S / nthreads; j++) {
	int y = (j * nthreads + tid) / L;
	int x = (j * nthreads + tid) % L;
	s[y, x] = A[k + y, yblock * L + x];
}
```

### More GPU optimization techniques

- Global memory continuous read

- Shared memory bank conflict

- Software pipelining

- Wrap level optimizations

- Tensor Core