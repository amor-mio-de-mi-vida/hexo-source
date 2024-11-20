---
date: 2024-11-20 09:36:40
date modified: 2024-11-20 11:24:34
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
$$Ya=b$$
$$
Y\in\mathbb{R}^{n\times d}, a\in\mathbb{R}^{d\times 1},b\in\mathbb{R}^{d\times 1}
$$
$$Y^TY\in\mathbb{R}^{d\times d}$$
$$Y^TYa=Y^Tb$$
