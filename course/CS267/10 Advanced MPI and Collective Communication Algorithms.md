---
date: 2024-11-20 09:36:40
date modified: 2024-12-03 09:42:25
title: Advanced MPI and Collective Communication Algorithms
tags:
  - course
categories:
  - cs267
---
**Collective Data Movement**

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.m3p7st5h.webp)

All collective operations must be called by *all* processes in the communicator.

`MPI_Bcast` is called by both the sender (called the root process) and the processes that are to receive the broadcast.

- "root" argument is the rank of the sender; this tells MPI which process originates the broadcast which receive.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.32hzqfz4sc.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4xukj2c3x9.webp)

- Many Routines: `Allgather`, `Allgatherv`, `Allreduce`, `Alltoall`, `Alltoallv`, `Bcast`, `Gather`, `Gatherv`, `Reduce`, `Reduce_scatter`, `Scan`, `Scatter`, `Scatterv`

- `All` versions deliver results to all participating process, not just root.

- `V` versions all the chunks to have variable sizes

- `Allreduce`, `Reduce`, `Reduce_scatter`, and `Scan` take both built-in and user-defined combiner functions.

**MXX: An MPI productivity library for C++11**

A few annoying redundancies:

- Any irregular exchange (e.g, `alltoallv`, `allgatherv`) is a multi step process: **(1) exchange counts**, (2) copy data to buffer, (3) allocate space, **(4) exchange actual data**

- have to create a derived data type for any non-PDO data

- have to map user defined functions to MPI functions

```c++
// The MXX way:
// lets take some pairs and find the one with the max second element
std::pair<int, double> v = ...;
std::pair<int, double> min_pair = mxx::allreduce(v, [](const std::pair<int, double>& x, const std::pair<int, double>& y) {
	return x.second > y.second ? x : y;
});
```

Available at: https://github.com/patflick/mxx

## SUMMA Algorithm

SUMMA = Scalable Universal Matrix Multiply, slightly less efficient than Cannon but is simpler and easier to generalize. 

Presentation from van de Geijn and Watts (Available at: https://github.com/patflick/mxx). Used in practice in PBLAS = Parallel BLAS. (Available at: https://github.com/patflick/mxx)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.5fkm7ocfie.webp)

`I`, `J` represent all rows, columns owned by a processor

`k` is a single row or column or a block of b rows or columns

$$C(I,J)=C(I,J)+\underset{k}{\sum}A(I,k)*B(k,J)$$

Assume a $p_r$ by $p_c$ processor grid ( $p_r=p_c=4$ above), Need not be square.

```cpp
int MPI_Comm_split( MPI_Comm comm,
					int color,
					int key,
					MPI_Comm *newcomm);
```

MPI's internal Algorithm:

- Use `MPI_Allgather` to get the color and key from each process

- Count the number of process with the same color; create a communicator with that many processes. If this process has `MPI_UNDEFINED` as the color, create a process with a single member.

- Use key to order the ranks.

- color controls assignment to new communicator, key controls rank assignment within new communicator.

```cpp
void SUMMA(double* mA, double* mB, double* mc, int p_c) {
	int row_color = rank / p_c; // p_c = sqrt(p) for simplicity
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
	
	int col_color = rank % p_c;
	MPI_Comm col_com;
	MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);
	
	for (int k = 0; k < p_c; k++) {
		if (col_color == k) memcpy(Atemp, mA, size);
		if (row_color == k) memcpy(Btemp, mB, size);
		
		MPI_Bcast(Atemp, size, MPI_DOUBLE, k, row_comm);
		MPI_Bcast(Btemp, size, MPI_DOUBLE, k, col_comm);
		
		SimpleDGEMM(Atemp, Btemp, mc, N/p, N/p, N/p);
	}
}
```

**MPI Built-in Collective Computation Operations**

| name         | usage                |
| ------------ | -------------------- |
| `MPI_MAX`    | Maximum              |
| `MPI_MIN`    | Minimum              |
| `MPI_PROD`   | Product              |
| `MPI_SUM`    | Sum                  |
| `MPI_LAND`   | Logical and          |
| `MPI_LOR`    | Logical or           |
| `MPI_LXOR`   | Logical exclusive or |
| `MPI_BAND`   | Binary and           |
| `MPI_BOR`    | Binary or            |
| `MPI_BXOR`   | Binary exclusive or  |
| `MPI_MAXLOC` | Maximu and location  |
| `MPI_MINLOC` | Minimum and location |
I specifically mention MPI as it enforces **certain semantic rules** (which also means that you can reimplement your own AllReduce if you have more relaxed semantics)

Example: **MPI_AllReduce**

- All processes must receive the same result vector:

- Reduction must be performed in canonical order $m_0+m_1+...+m_{p-1}$ (if the operation is not commutative)

- The same reduction order and bracketing for all elements of the result vector is note strictly required, but should be strived for.

| Communication   | Latency                       | Bandwidth              | Computation            |
| --------------- | ----------------------------- | ---------------------- | ---------------------- |
| Broadcast       | $\lceil\log_2(p)\rceil\alpha$ | $n\beta$               | ——                     |
| Reduce(-to-one) | $\lceil\log_2(p)\rceil\alpha$ | $n\beta$               | $\frac{p-1}{p}n\gamma$ |
| Scatter         | $\lceil\log_2(p)\rceil\alpha$ | $\frac{p-1}{p}n\beta$  | ——                     |
| Gather          | $\lceil\log_2(p)\rceil\alpha$ | $\frac{p-1}{p}n\beta$  | ——                     |
| Allgather       | $\lceil\log_2(p)\rceil\alpha$ | $\frac{p-1}{p}n\beta$  | ——                     |
| Reduce-scatter  | $\lceil\log_2(p)\rceil\alpha$ | $\frac{p-1}{p}n\beta$  | $\frac{p-1}{p}n\gamma$ |
| Allreduce       | $\lceil\log_2(p)\rceil\alpha$ | $2\frac{p-1}{p}n\beta$ | $\frac{p-1}{p}n\gamma$ |
Note: Pay particular attention to the conditions for the lower bounds given in the text

**AllGather**

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.39l84f9yzu.webp)

- At time t: send the data you received at time t-1 to your right, and receive new data from your left.

- At time 0, send your original data

- Optimal bandwidth, high latency

- <mark>Not as bad as it sounds if pipelined</mark> (NCCL exclusively uses piplined ring algorithms for its collectives)

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.361m6pjwgu.webp)

- **At time t**: process i exchanges (send/recv) all its current data (its original data plus anything received until then) with process $i\pm 2^t$

- Data exchanged at each step: $n/p$, $2n/p$, $4n/p$, ..., $2^{lg(p)-1}n/p$

- Tricky for non-power-of-two.


AllGather—— The Bruck Algorithm
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4g4jd14t4f.webp)

- **At time t**: process $i$ receives all your current data from process $i\pm 2^t$ and sends all of its current data to process $i-2^t$ (both modulo p)

- This regular exchange ends after $\lfloor lg(p)\rfloor$ steps

- At the last communication step, instead of receiving/sending all current data, send/recv only the top ($p-2^{\lfloor lg(p)\rfloor}$) entries

- Requires a final, local shift to get data in the correct order.

- For any p: $T_{brock}=\alpha\lceil lg(p)\rceil+\beta n(p-1)/p$

- By contrast, recursive doubling takes $2\lfloor lg(p)\rfloor$ steps for non-power-of-two processor counts.

Similar ideas are used in other collectives (e.g. **recursive halving** instead of recursive doubling for **reduce-scatter**) with different local computations (e.g. for **all-reduce**, perform a **local reduction** at each step instead of **concatenating** data as in all-gather)

### Synchronization

- `MPI_Barrier` (comm)

- Blocks until all processes in the group of the communicator `comm` call it

- Almost never required in a parallel program (Occasionally useful in measuring performance and load balancing)

### Nonblocking Collective Communication

Nonblocking variants of all collectives

- `MPI_lbcast(<bcast args>, MPI_Request *req);`

Semantics:

- Function returns no matter what 

- **No guaranteed progress (quality of implementation)**

- Usual completion calls (wait, test) + mixing

- Out-of-order completion

Restrictions:

- No tags, in-order matching

- Send and vector buffers may not be touched during operation

- **No matching with blocking collectives**

**Semantic advantages**

- Enable asynchronous progression (and manual, Software pipelining)

- Decouple data transfer and synchronization (Noise resiliency!)

- Allow overlapping communicators (See also neighborhood collectives)

- Multiple outstanding operations at any time (Enables pipelining window)


```cpp
// SUMMA in MPI

void SUMMA(double *mA, double *mB, double *mc, int p_c) {
	int row_color = rank / p_c; // p_c = sqrt(p) for simplicity
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
	
	int col_color = rank % p_c;
	MPI_Comm col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);
	
	double *mA1, *mA2, *mB1, *mB2;
	colsplit(mA, mA1, mA2); // split mA by the middle column
	rowsplit(mB, mB1, mB2); // split mB by the middle row
	
	if (col_color == 0) memcpy(Atemp1, mA1, size);
	if (row_color == 0) memcpy(Btemp1, mB1, size);
	
	MPI_Request reqs1[2];
	MPI_Request reqs2[2];
	MPI_Ibcast(Atemp1, size, MPI_DOUBLE, k, row_comm, &reqs1[0]);
	MPI_Ibcast(Btemp1, size, MPI_DOUBLE, k, col_comm, &reqs1[1]);
	
	for (int k = 0; i < p_c - 1; k++) {
		if (col_color == k) memcpy(Atemp2, mA2, size);
		if (row_color == k) memcpu(Btemp2, mB2, size);
		
		MPI_Ibcast(Atemp2, size, MPI_DOUBLE, k, row_comm, &reqs2[0]);
		MPI_Ibcast(Btemp2, size, MPI_DOUBLE, k, col_comm, &reqs2[1]);
		
		MPI_Waitall(reqs1, MPI_STATUS_IGNORE);
		SimpleDGEMM(Atemp1, Btemp1, mC, N/p, N/p, N/p);
		
		if (col_color == k) memcpy(Atemp1, mA1, size);
		if (row_color == k) memcpy(Btemp1, mB1, size);
		
		MPI_Ibcast(Atemp1, size, MPI_DOUBLE, k, row_comm, &reqs1[0]);
		MPI_Ibcast(Btemp1, size, MPI_COUBLE, k, col_comm, &reqs1[1]);
		
		MPI_Waitall(reqs2, MPI_STATUS_IGNORE);
		SimpleDGEMM(Atemp2, Btemp2, mC, N/p, N/p, N/p);
	}
	
	if (col_color == p-1) memcpy(Atemp2, mA2, size);
	if (row_color == p-1) memcpy(Btemp2, mB2, size);
	
	MPI_Ibcast(Atemp2, size, MPI_DOUBLE, k, row_comm, &reqs2[0]);
	MPI_Ibcast(Btemp2, size, MPI_DOUBLE, k, col_comm, &reqs2[1]);
	
	MPI_Waitall(reqs1, MPI_STATUS_IGNORE);
	SimpleDGEMM(Atemp1, Btemp1, mC, N/p, N/p, N/p);
	
	MPI_Waitall(reqs2, MPI_STATUS_IGNORE);
	SimpleDGEMM(Atemp2, Btemp2, mC, N/p, N/p, N/p);
}
```

## Hybrid Programming with Threads

