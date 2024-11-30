---
date: 2024-11-04 07:15:55
date modified: 2024-11-26 10:46:45
title: Matrix Multiplication and  the Roofline Model
tags:
  - course
categories:
  - cs267
date created: 2024-09-25 13:26:08
---
## Matrix Multiplication

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

### Naive Matrix Multiply

``` 
{implements C = C + A * B}
for i = 1 to n
	{read row i of A into fast memory}
	for j = 1 to n
		{read C[i,j] into fast memory}
		{read column j of B into fast memory}
		for k = 1 to n
			C[i,j] = C[i,j] + A[i,k] * B[k,j]
```

- $n^2$ to read each row of A once

- $2n^2$ to read and write each element of C once

- $n^3$ to read each column of B n times

$f=2n^3$ arithmetic ops. $m=n^3+3n^2$ slow memory

So the computational intensity is:

$CI=f/m=2n^3/n^3+3n^2\approx 2$

No better than matrix vector!

### Blocked (Tiled) Matrix Multiply

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.7paxnmkkx.webp)

The idea is if I can fit three of these blue blocks into fast memory at the same time, then there's no more memory traffic when I do this little B by B matrix multiply while it's inside the cache.

Consider A, B, C to be N-by-N matrix of b-by-b sub-blocks where $b=n/N$ is called the block size

```
for i = 1 to N
	for j = 1 to N
		{read block C(i,j) into fast memory}
		for k = 1 to N
			{read block A(i,k) into fast memory}
			{read block B(k,j) into fast memory}
			C(i,j)=C(i,j)+A(i,k)*B(k,j) 
		{write block C(i,j) back to slow memory}
```

- $2n^2$ read and write each block of C once ($2N^2*b^2=2n^2$)

- $N*n^2$ to read each block of A $N^3$ times ($N^3b^2=N^3*(n/N)^2$)

- $N*n^2$ to read each block of B $N^3$ times ($N^3b^2=N^3*(n/N)^2$)

words moved = $m = 2n^2+N*n^2+N*n^2=(2N+2)*n^2$

Computational Intensity = $CI = f/m = 2n^3((2N+2)*n^2)\approx n/N=b$ for large n

Assume our fast memory has size M:

$b\le \sqrt{M/3}$

Since M must hold 3 $b\times b$ blocks.

### Recursive Matrix Multiplication

```
def RMM(A, B, n):
	if n=1:
		C = A * B
	else:
		C_00 = RMM(A_00, B_00, n/2) + RMM(A_01, B_10, n/2)
		C_01 = RMM(A_00, B_01, n/2) + RMM(A_01, B_11, n/2)
		C_10 = RMM(A_10, B_00, n/2) + RMM(A_11, B_10, n/2)
		C_11 = RMM(A_10, B_01, n/2) + RMM(A_11, B_11, n/2)
	return C
```

$$\begin{align}
\text{Arith}(n)
&=\text{\# arithmetic operations in RMM(.,.,n)}\\
&= 8\cdot \text{Arith}(n/2) + 4(n/2)^2\text{ if n > 1, else 1}\\
&=2n^3-n^2
\end{align}$$
$$\begin{align}
W(n)
&=\text{\# words moved between fast, slow memory by RMM(.,.,n)}\\
&=8\cdot W(n/2)+ 4\cdot 3(n/2)^2\text{if n>1, else 1}\\
&=O(n^3/(M_{fast})^{1/2}+n^2)

\end{align}$$

### Alternate Data Layouts

- May also use blocked or recursive layouts

- Several possible recursive layouts, depending on the order of the sub-blocks

- Copy optimization may be used to move

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4n7q2xxrye.webp)

{% note primary %}
Theorem (Hong & Kung, 1981):
Any reorganization of matmul (using only commutativity and associativity) has computational intensity $q=O((M_{fast})^{1/2})$, so \# words moved between fast/slow memory = $\Omega(n^3/(M_{fast})^{1/2})$  
{% endnote %}

- Parallel matrix multiply, optimize latency as well as bandwidth

- Rest of linear algebra (Gaussian elimination, least squares, tensors ...)

- Nested loops accessing arrays (eg All-Pairs-Shortest-Paths, N-body)

Open problems:

- Small loop bounds (eg matrix-vector vs matrix-matrix multiply)

- Dependencies, i.e. when only some reorganizations are correct.

Because all of these reorganizations depend on the fact that the algorithm gives us a lot of freedom to do the operations in a different order.


### Strassen's Matrix Multiply

![[Pasted image 20241104080856.png]]

$$\begin{align}
T(n)&=\text{Cost of multiplying nxn matrices}\\
&=7*T(n/2)+18*(n/2)^2\\
&=O(n^{\log_27})\\
&=O(n^{2.81})
\end{align}$$
basic linear algebra subroutines (BLAS)

Industry standard interface: www.netlib.org/blas, www.netlib.org/blas/blast--forum

LINPACK benchmark

## Roofline Model —— How fast can an algorithm go in practice?

Understand performance behavior

- Differences between Architectures, Programming Models, implementations, etc.

### Roofline

Idea: applications are limited by either compute peak or memory bandwidth:

- Bandwidth bound (matvec)

- Compute bound (matmul)

What is in the Roofline Model?

**Arithmetic performance (flop/sec)**

- Clock Speed and Parallelism (ILP, SIMD, Multicore)

**Memory bandwidth (bytes/sec)**

- Latency not included (looking at best case)

**Computational (Arithmetic) Intensity**

- Application balances (flops/word or flops/byte)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.2objcojgey.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.7i0e8w6osn.webp)

Assume

- Idealized processor/caches

- Code start (data in DRAM)

$$
\text{Time}=\max\cases{
\text{\#FP ops / Peak GFLOP/s}\\
\text{\#Bytes / Peak GB/s}
}
$$

$$\text{GFlop/sec}=\min
\cases{\text{Peak GFLOP/s}\\
\text{(\#FP ops / \#Bytes) * Peak GB/s}
}$$

the machine balance is how many flops per byte do you have to do so that you hit the peak speed.

Bandwidth Bound and Compute Bound

There are multiple roofs depending on which parts of the architecture you can take advantage of 

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.54xrroydnu.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.6f0oy0hrfl.webp)

Roofline captures upper bound performance

The min of 2 upper bounds for a machine

- Peak flops (or other arith ops)

- Memory bandwidth max

Algorithm computational intensity

- Usually defined as best case, infinite cache

Originally for single processors and SMPs

Widely used in practice and adapted to any bandwidth/compute limit situation.

Are you limited by bandwidth or compute? how much performance have you left on the table? How much improvements are potential.


