---
date: 2024-11-10 08:43:59
date modified: 2024-11-10 17:27:34
title: More on Communication-optimal Matmul (and beyond)
tags:
  - course
categories:
  - cs267
---
Communication = moving data

- Between main memory and cache

- Between processors over a network

- Most expensive operation (in time or energy)

Goal: Provably minimize communication for algorithms that look like nested loops accessing arrays

- Includes matmul, linear algebra (dense and sparse), n-body, convolutional neural nets (CNNs), ...

Simple case: n-body (sequential, with main memory and cache)

- Communication lower bound and optimal algorithm

Extension to Matmul

Extension to algorithms that look like nested loops accessing arrays, like CNNs (and open questions)

<!-- more -->

### Data access for n-body

A() = array of structures

- A(i) contains position, charge on particle i

Usual n-body

- `for i=1:n, for j=1:n except i, F(i) = F(i) + force(A(i), A(j))`

Simplify to make counting easier

- Let B() = array of disjoint set of particles

- `for i = 1:n, for j = 1:n, e = e + potential(A(i), B(j))`

Simplify more

- `for i = 1:n, for j = 1:n, access A(i) and B(j)`

 ![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.92q5h8v3mh.webp)
### Communication lower bound for n-body (intuition)

`for i=1:n, for j=1:n, access A(i), B(j)`

With a cache of size M full of data, can only perform $M^2/4$ loop iterations

To perform all $n^2$ loop iterations, need to (re)fill cache $n^2/(M^2/4)=4(n/M)^2$ times.

Filling cache costs M reads from slow memory

Need to do at least $4(n/M)^2*M=4n^2/M$ reads

- Can improve constant slightly

- Write as $\Omega(n^/M)=\Omega(\#\text{loop iterations}/ M)$

### Generalizing to other algorithms

Many algorithms look like nested loops accessing arrays

- Linear Algebra (dense and sparse)

- Grids (structured and unstructured)

- Convolutional Neural Nets (CNNs) ...

Matmul: $C = A * B$

`for i = 1:n, for j=1:n, for k=1:n`, `C(i,j)=C(i,j)+A(i,k)*B(k,j)`

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.39l77zndyn.webp)

**loop iterations** doable with M words of data = **cubes** $\le$ (area(A shadow) $\cdot$ area(B shadow) $\cdot$ area(C shadow))$^{1/2}\le (M\cdot M\cdot M)^{1/2}=M^{3/2}=F$

Need to read/write at least M $n^3/F=\Omega(n^3/M^{1/2}=\Omega(\#\text{loop iterations} / M^{1/2})$ words to/from cache.

Analogous to n-body:

- What is the largest set of `C(i,j)+=A(i,k)*B(k,j)` we can perform given M entries `A(i,k)`, `B(k,j)`, `C(i,j)`?

- What is the largest set of `(i,j,k)` we can have, given a bound M on the number of `(i,k)`, `(k, j)`, `(i,j)` ?

- What is the shape of the largest 3D volume we can have, given a bound M on the area of its shadows in 3 directions?

- Answer: A cube, with edge length $O(M^{1/2})$, volume $O(M^{3/2}$

- Optimal "blocked" Algorithm: 6 nested loops, 3 innermost loops do $b\times b$ matmul with $b=O(M^{1/2})$

**Parallel case:** apply reasoning to one processor out of P

- "Fast memory" = local processor, "Slow memory" = other processor

- Goal: lower bound \# "reads/writes" = \# words moved between one processor and others.

- \# loop iterations = $n^3/P$ (load balanced)

- $M=3n^2/P$ (each processor gets equal fraction of data)

- $\textbf{reads/writes}\ge M\cdot (n^3/P)/(M)^{3/2}=\Omega(n^2/P^{1/2})$

### Approach to generalizing lower bounds

**Matmul**

```pseudo code
### from
for i = 1:n
	for j = 1:n
		for k = 1:n
			C(i,j)+=A(i,k)*B(k,j)

### to
for (i,j,k) in S = subset of Z^3
	Access locations indexed by (i,j), (i,k), (k,j)
```

**General case**

```pseudo
### from 
for i1 = 1:n
	for i2 = i1:m
		...
			for ik = i3:i4
				C(i1+2*i3-i7) = func(A(i2+i3*i4,i1,i2,i1+i2,...),B(pnt(3*i4),...))
				D(something else) = func(something else),...

### to
for (i1,i2,...,ik) in S = subset of Z^k
	Access locations indexed by "projections", e.g.
	ø_C(i1,i2,...,ik)=(i1+2*i3-i7)
	ø_A(i1,i2,...,ik)=(i2+3*i4,i1,i2,i1+i2,...),...
```
Goal: Communication lower bounds, optimal algorithms for *any* program that looks like this.

{% note primary %}
**Theorem**
Given a program with array refs given by projections $\phi_j$, then there is an $s_{HBL}\ge 1$ such that 
$$
\textbf{words\_moved} = \Omega(\textbf{iterations}/M^{s_{HBL}-1})
$$
where $s_{HBL}$ is the value of a linear program:
- minimize $s_{HBL}=\sum_je_j$ subject to
- $rank(H)\le\sum_je_j*rank(\phi_j(H))$ for all subgroups $H<Z^k$
{% endnote %}   

Proof depends on recent result in pure mathematics by Christ/Tao/Carbery/Benneet

- Generalization of Holder-Brascamp-Lieb (HBL) inequality to Abelian groups

- HBL generalizes Cauchy-Schwartz, Loomis-Whitney, ...

### Is this bound attainable?

{% note primary %}
**Theorem**
We can always construct an optimal tiling, that attains the lower bound
{% endnote %}   

**Assumptions/caveats/open questions**

Attains lower bound $\Omega(\textbf{iterations}/M^{s_{HBL}-1})$ in $O()$ sense

Depends on loop dependencies

- Not all tilings may compute the right answer

- Best case: no dependencies, or just reductions (like matmul)

Assumes loop bounds are large enough to fit tile

- Ex: same lower bound for matmul applies to matrix-vector-multiply, but not attainable

- Recent extension to arbitrary loop bounds, assuming all subscripts "projective" e.g. (i), (i, j), (i, j, k) etc.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.5j47riwsgd.webp)

### CNN using Im2col

- Same operations as 7 nested loops

- Can exploit optimized matmul

- Need to replicate data

- Can we communicate less, by doing convolutions directly?

### Communication Lower Bound for CNNs

Let $N=\textbf{iterations}=KHWRSCB, M=\text{cache size}$

{% note primary %}
$$
\begin{align}\textbf{words moved}=\Omega&(\max(\quad\text{ ... 5 terms}\\
&BKHW, \quad\text{ ... size of Out}\\
&\sigma_H\sigma_WBCWH, \quad\text{ ... size of Image}\\
&CKRS, \quad\text{ ... size of Filter}\\
&N/M, \quad\text{ ... same lower bound as n-body}\\
& N/(M^{1/2}(RS/(\sigma_H\sigma_W))^{1/2}), \quad\text{ ...new lower bound}))
\end{align}$$
If your filters are large enough, you can beat matrix multiply a lot.
{% endnote %}    

New lower bound

- Beats matmul by factor $(RS/(\sigma_H\sigma_W))^{1/2}$

- Applies in common case when data does not fit in cache, but one $R\times S$ does

- Tile needed to attain $N/M$ too big to fit in loop bounds

Attainable (many cases, solved using Mathematical)