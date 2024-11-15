---
title: Data Parallel Algorithms
tags:
  - course
categories:
  - cs267
date modified: 2024-11-14 19:18:11
date: 2024-11-13 18:14:09
---
Data parallelism: perform the same operation on multiple values (often array elements)

- Also includes reductions, broadcast, scan...

Many parallel programming models use some data parallelism.

- SIMD units (and previously SIMD supercomputers)

- CUDA/GPUs

- MapReduce

- MPI collectives
<!-- more -->

## Data Parallel Patterns

**Data Parallel Programming: Unary Operators**

- Unary operations applied to all elements of an array
```
A = array
B = array
f = square (any unary function, i.e., 1 argument)
B = f(A)
```

**Data Parallel Programming: Binary Operators**

- Binary operations applied to all pairs of elements
```
A = array
B = array
C = array
- or any other binary operator
C = A - B
```

**Data Parallel Programming: Broadcast**

- Broadcast fill a value into all elements of an array
```
a = scalar
B = array

B = a // broadcast

X = array
Y = array
Z = array

Z = a*X + Y // axpy
```

Useful for `a*X+Y`, called `{s,d}` axpy

For single, double precision, or in general

**Memory Operations: Strided and Scatter/Gather**

- Array assignment works if the arrays are the same shape

```
A: double [0:4]
B: double [0:4] = [0.0, 1.0, 2.2, 3.3, 4.4]

A = B
```

- May have a stride, i.e., not be contiguous in memory

```
A = B[0:4:2] // copy with stride 2 (every other element)
C: double [0:4, 0:4]
A = C[*,3] // copy column of C
```

- Gather (indexed) values, from one array

``` 
X: int[0:4] = [3,0,4,2,1] // a permutation of indices 0 to 4
A = B[X] // A now is [3.3,0.0,4.4,2.2,1.1]
```

- Scatter (indexed) values from one array

```
A[X] = B // A now is [1.1,4.4,3.3,0.0,2.2]
```

> What if `X=[0,0,0,0,0]`?
> 
> That's going to assign the first entry of B to $a_0$, and the second entry of B to $a_0$, and so forth, so this is a natural race condition, or an error and basically your system should not, no promises are made if you do this, this is a program error.
> 
> So you have to make sure that you don't have any common destinations like that.
 
**Data Parallel Programming: Masks**

- Can apply operations under a "mask"
```
M = array of 0/1 (True/False)
A = array
B = array

A = A + B under M
```

> That also lets us do a lot of operations with branches and so forth, I can have conditionals 

**Data Parallel Programming: Reduce**

- Reduce an array to a value with + or any associative op
```
A = array
b = scalar
b = sum(A)
```
- Associative so we can perform op in different order

- Useful for dot products (ddot, sdot, etc.)
```
b = dot(X.T, Y) 
```

> This is not done in one time step, Obviously, there's more work and it's going to take $\log n$ steps in order to sum up n numbers.
> 
>  In order for this to work correctly, you have to make an assumption in order to get the right answer. **Your operator has to be associative.** That means you're allowed to do the operations in any order you like, as long as you maintain the original order, you can put the parentheses wherever you like and get the same answer .

**Data Parallel Programming: Scans**

- Fill array with partial reductions any associative op

- Sum scan
```
A = array
B = array
B = scan(A,+)
```

- Max scan:
```
B = scan(A,max)
```

> It let us do this surprising stuff like invert matrices log squared in time. It seems like this is an inherently serial operation. I got to do it from left to right. But you can do it in log n time. There is a very clever algorithm to do this.

> Inclusive and Exclusive Scans
> 
> Two variations of a scan, given an input vector $[x_0,x_1,...,x_{n-1}]:$
> 
> - **inclusive** scan includes input $x_i$ when computing output $y_i$
> $$[a_0,(a_0\odot a_1),\cdots,(a_0\odot a_1\cdots\odot a_{n-1})]$$
> e.g., `add_scan_inclusive([1,0,3,0,2])`$\rightarrow$`1,1,4,4,6`
> 
> - **exclusive** scan does *not* $x_i$ when computing output $y_i$
> $$[I,a_0,(a_0\odot a_1),\cdots,(a_0\odot a_1\cdots\odot a_{n-2})]$$ where $I$ is the identity for $\odot$
> 
> e.g., `add_scan_exclusive([1,0,3,0,2])`$\rightarrow$`[0,1,1,4,4]`
> 
> Can easily get the inclusive version from the exclusive, for the other way you need an inverse for the operator (or shift)

## Idealized Hardware and Performance Model

### SIMD Systems Implemented Data Parallelism

SIMD Machine: A large number of (usually) tiny processors.

- A single "control processor" issues each instruction.

- Each processor executes the same instruction.

- Some processors may be turned of in some instructions.

Machine

- An unbounded number of processors (p)

- Control overhead is free

- Communication is free

Cost (complexity) on this abstract machine is the algorithm's span or depth , $T_\infty$

- Define a lower bound on time on real machines.

### Cost on Ideal Machine

**Span for unary or binary operations (pleasingly parallel)**

- Cost $O(1)$ since p is unbounded

- Even if arrays are not aligned, communication is "free" here.

**Reductions and broadcasts**

- Cost $O(log(n))$, using a tree of processors.

 - Interestingly, there's a very simple proof that that's a lower bound. No matter how you do broadcast or reductions, as long as you only have binary operations, it's going to take at least log n time.

 - And this is why I need associatively because implicitly you're not summing the numbers from left to right, I'm putting parentheses, in a nested fashion around them.
 
>  **Lower bound proof:**
>  
>  - Given a function $f(x_1,...,x_n)$ of n input variables and 1 output variable, how fast can we evaluate it in parallel
>  
>  - Assume we only have binary operations, one per time step.
>  
>  - After 1 time step, an output can only depend on two inputs.
>  
>  - By induction: after k time units, an output can only depend on $2^k$ inputs. In other words, after $\log_2n$ time units, output depends on at most n inputs.
>  
>  - A binary tree performs such a computation.

> **Multiplying n-by-n matrices in $O(\log n)$ time**
> 
> **Step 1**: For all $1\le i,j,k\le n$, $P(i,j,k)=A(i,k)*B(k,j)$
> 
> - cost = 1 time unit, using $n^3$ processors
> 
> **Step 2**: For all $1\le i,j\le n$, $C(i,j)=\overset{n}{\underset{k=1}{\sum}}P(i,j,k)$
> 
> - cost = $O(\log n)$ time, using $n^2$ trees, $n^3/2$ processors each
> 
> You can't imagine multiplying matrices faster than log n time just because the dependencies.

**What about Scan (aka Parallel Prefix)**

Recall: the **scan** operation takes a **binary associative** operator $\odot$, and an array of n elements 
$$[a_0,a_1,a_2,...,a_n]$$
and produces the array
$$[a_0,(a_0\odot a_1),\cdots,(a_0\odot a_1,...,\odot a_{n-1})]$$
It looks like this:

```
y[0]=0
for i = 1 ... n:
	y[i] = y[i-1] + x[i]
```

Takes n-1 operations to do in serial. The $\textbf{i}^{\textbf{th}}$ iteration of the loop depends completely on the $(\textbf{i-1})^{\textbf{st}}$ iteration.

> Naive version: we can use **reduction** to do this, put 1 processor at element 1, 2 at element 2, 3 at position 3. ($O(\log n)$ span 🙂, $O(n^2)$ work ☹️)

> **Sum Scan (aka prefix sum) in parallel**
> Algorithm: 1. Pairwise sum 2. Recursive prefix 3. Pairwise sum 
> ![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.pfczwx0s7.webp)
> Time for this algorithm on one processor (work)
> 
> - $T_1(n)=\frac{n}{2}+\frac{n}{2}+T_1(n/2)=n+T_1(n/2)=2n-1$
> 
> Time on unbounded number of processors (span)
> 
> - $T_\infty(n)=2\log n$
> 
> Now the issue with this is that it sort of seems like I need a lot of extra memory in order to keep track of all these intermediate results. it turns out, you can un this algorithm in place. You can just overwrite the array after log n steps it's replaced by parallel prefix so you don't need any extra memory to run this.
> 
> ![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.92q5lutf33.webp)
> 
> It took only $O(2\log n)$ steps and we only did $O(2n)$ additions. It is entirely in place, I didn't need any extra memory in order to do this.

## (Non-trivial) Applications of Data Parallelism

### Scans are useful for many things (partial list here)

- Adding two n-bit integers in $O(\log n)$ time.

- Inverting n-by-n triangular matrices in $O(\log^2 n)$ time

- Inverting n-by-n dense matrices in $O(\log^2 n)$ time

- Evaluating arbitrary expressions in $O(\log n)$ time

- Evaluating recurrences in $O(\log n)$ time

- "2D parallel prefix", for image segmentation (Catanzaro & Keutzer)

- Sparse-Matrix-Vector-Muliply (SpMV) using Segmented Scan

- Parallel page layout in a browser (Leo Meyerovich, Ras Bodik)

- Solving n-by-n tridiagonal matrices in $O(\log n)$ time

- Traversing linked lists

- Computing minimal spanning trees

- Computing convex hulls of point sets ...

### Application: Stream Compression

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.7p3mhtypnr.webp)

By a mixture of using masking, indirect indexing, I can pick any subset I want and get that to happen in principle in one operation.


**Application: Remove elements satisfying a condition**

Given an array of values, and an index x, remove all elements that are not divisible by x: 

```
int find(int x, int y) (y % x == 0) ? 1 : 0;
```

Use previous solution to remove those not divisible.

### Application: Radix Sort (serial algorithm)

- Idea: Sort 1 bit at a time, 0s on left, 1s on right.

- Use a "stable" sort: Keep order as it, unless things need to switch based on the current bit.

- Start with least-significant bit, and move up.

 And so the number of steps, if I can do each one of those sorting steps in a constant amount of time, the number of steps is just the number of bits in all the numbers. And so the question is, **how can I do one of these sorting steps in a constant amount of time using all these operations?**

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.9gwlcrenmt.webp)

```
input                                         4 7 2 6 3 5 1 0
odds = last bit of each element               0 1 0 0 1 1 1 0
evens = complement of odds (last bit = 0)     1 0 1 1 0 0 0 1
evpos = exclusive sum scans of evens          0 1 1 2 3 3 3 3 4
totalEvens = broadcast last element           4 4 4 4 4 4 4 4
indx = constant array of 0...n                0 1 2 3 4 5 6 7
oddpos = totalEvens + indx-evpos              4 4 5 5 5 6 7 8
pos = if evens then evpos else oddpos         0 4 1 2 5 6 7 3
Scatter input using pos as index              4 7 2 6 3 5 1 0
Repeat with next bit to left until done       4 2 6 0 7 3 5 1
```
### List Ranking with Pointer Doubling

Given a linked list of N nodes, find the distance (\# hops) from each node to the end of the list.

```
d(n) = 
	0 if n.next is null
	1 + d(n.next) otherwise

val = 1
while next != null
	val += next.val
	next = next.next
	 
// Works if nodes are on arbitary processors.
```

Approach: put a processor at every node

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.6pnj5xxhcp.webp)

### Application: Fibonacci via Matrix Multiply Prefix

We can do arbitrary linear recurrences in $\log n$ time.

$$F_{n+1}=F_n+F_{n-1}$$
$$\begin{align}
\left(\begin{matrix}F_{n+1}\\F_n\end{matrix}\right)=
\left(\begin{matrix}1&1\\1&0\end{matrix}\right)
\left(\begin{matrix}F_n\\F_{n-1}\end{matrix}\right)
\end{align}$$
Can compute all $F_n$ by matmul_prefix on 
$$[\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right),\left(\begin{matrix}1&1\\1&0\end{matrix}\right)]$$

then select the upper left entry.

Same idea works for any linear recurrence.

### Application: Adding n-bit integers in $O(\log n)$ time

Computing sum s of two n-bit binary numbers, think of a and b as array of bits

`a=a[n-1]a[n-2]...a[0]` and `b=b[n-1]b[n-2]...b[0]`

`s=a+b=s[n]s[n-1]...s[0]` (use carry-bit array `c=c[n-1]c...c[0]c[-1]`)
 
```
c[-1] = 0            ... rightmost carrybit
s[i] = (a[i] xor b[i]) xor c[i-1]  ... one or three 1s
c[i] = (a[i] xor b[i] and c[i-1]) or (a[i] and b[i]) ... next carry bit

for all (0<=i<=n-1) p[i]=a[i] xor b[i] ... propagate bit
for all (0<=i<=n-1) g[i]=a[i] and b[i] ... generate bit
```

**Challenge**: compute all `c[i]` in `O(log n)` time via parallel prefix.

$$\begin{align}\left(\begin{matrix}c[i]\\1\end{matrix}\right)&=
\left(\begin{matrix}p[i]\text{ and }c[i-1]\text{ or }g[i]\\1\end{matrix}\right)\\&=
\left(\begin{matrix}p[i]&g[i]\\0&1\end{matrix}\right)
\left(\begin{matrix}c[i-1]\\1\end{matrix}\right)\\&=
M[i]*\left(\begin{matrix}c[i-1]\\1\end{matrix}\right)\\
&=M[i]*M[i-1]*...*M[0]*\left(\begin{matrix}0\\1\end{matrix}\right)
\end{align}$$
evaluate `M[i]*M[i-1]*...*M[0]` by parallel prefix

2-by-2 Boolean matrix multiplication is associative.

Every piece of hardware that has ever been built to do addition on the CPU uses this trick, and it's called **carry look ahead edition**

### Application: Lexical analysis (tokenizing, scanning)

Given a language of:

- Identifiers (Z): string of chars

- Strings (S): in double quotes

- Ops (\*): +,-,\*,=,<,>,<=,>=

- Expression(E), Quotes (Q), ...

Lexical analysis (Divide into tokens)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4ckwoqij56.webp)

- Each state in first column; N initial state.

- Each row gives the next state based on the next character at the top.

- Apply string Y"+ to state Z written as ZY"+=((ZY)")+=(Z")+=Q+=S.

- Each column is a state transition for that character.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4jo4k63kcv.webp)

**Lexical analysis **

- Replace every character in the string with the array representation of its state-to-state function (column).

- Perform a parallel-prefix operation with $\odot$ as the array composition. Each character becomes an array representing the state-to-state function for that prefix.

- Use initial state (N, row 1) to index into these arrays.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.3d4tbkdbh9.webp)

How do I do this pairwise combination of stuff to do everything in $\log n$ time? we can replace two columns with one column representing the pair of transitions. If you can imagine a reduction operation I could just do a tree of those and it would tell me what state I'm in at the end. The same trick works for parallel prefix because I want to know all the intermediate states, so I know where the tokens begin and end.

### Applications: Inverting triangular n-by-n matrices in $O(\log^2 n)$ Time

$$\left(\begin{matrix}A&0\\C&B\end{matrix}\right)^{-1}=
\left(\begin{matrix}A^{-1}&0\\-B^{-1}CA^{-1}&B^{-1}\end{matrix}\right)$$

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.491ar0gsrj.webp)

### Application: Inverting Dense n-by-n matrices in $O(\log^2 n)$ time

**Lemma 1: Cayley-Hamilton Theorem**

- Says that every matrix satisfies the polynomial and the coefficients of the polynomial are the characteristic polynomial whose roots would be the eigenvalues.

- Expression for $A^{-1}$ via characteristic polynomial in A

**Lemma 2: Newton's Identities**

- Triangular system of equations for coefficients of characteristic polynomial, where matrix entries = $s_k$

**Lemma 3:** $s_k=\text{trace}(A^k)=\overset{n}{\underset{i=1}{\sum}}A^k[i,i]$

Csanky's Algorithm

- Compute the powers $A^2$, $A^3$, ..., $A^{n-1}$ by parallel prefix, cost = $O(\log^ n)$

- Compute the traces $s_k=\text{trace}(A^k)$, cost = $O(\log n)$

- Solve Newton identities for coefficients of characteristic polynomial, cost = $O(\log^2 n)$

- Evaluate $A^{-1}$ using Cayley-Hamilton Theorem, cost = $O(\log n)$

**Completely numerically unstable**

### Segmented Scans

Inputs = value array, flag array, associative operator $\odot$

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.60u9lxtzqm.webp)

## Mapping Data Parallelism to Real Hardware

**SIMD/Vector Processor Use Data Parallelism**

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.6t753oj5ix.webp)

**Mapping to GPUs**

- For n-way parallelism may use n threads, divided into blocks

- Merge across statements (so A=B; C=A; is a single kernel)

- Mapping threads to ALUs and blocks to SMs is compiler/hardware problem

**Bottom Line** 

- Branches are still expensive on GPUs

- May pad with zeros / nulls etc. to get length

- Often write code with a guard (if i < n), which will turn into mask, fine if n is large

- Non-contiguous memory is supported, but will still have a higher cost

- Enough parallelism to keep ALUs busy and hide latency, memory/scheduling tradeoff

**Mapping Data Parallelism to SMPs (and MPPs)**

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.pfd18nidk.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.491ar1rkp3.png)

The $\log_2 n$ span is not the main reason for the usefulness of parallel prefix.

Say n = k \* p (k = 1,000,000 elements per proc)

```
cost = (k adds) + // compute and store k values a[0],...,a[k-1]
	   (log P steps) + // parallel scan on a[k-1] values
	   (k adds) // add 'my' scan result to a[0],...,a[k-1]
```

Key to implementing data parallel algorithms on clusters, SMPs, MPPs, i.e., modern supercomputers.

## Summary of Data Parallelism

**Sequential semantics (or nearly) is very nice**

- Debugging is much easier without non-determinism

- Correctness easier to reason about

**Cost model is independent of number of processors**

- How much inherent parallelism

**Need to "throttle" parallelism**

- n >> p can be hard to map, especially with nesting

- Memory use is a problem

**More reading**

- Classic paper by Hillis and Steele “Data Parallel Algorithms” https://doi.org/10.1145/7902.7903 and on Youtube

- Blelloch the NESL languages and “NESL Revisited” paper, 2006