---
date: 2024-11-16 14:32:01
date modified: 2024-11-17 15:15:49
title: Distributed Memory Machines and Programming
tags:
  - course
categories:
  - cs267
---
## Distributed Memory Architectures

### Properties of communication networks

Early distributed memory machines were:

- Collection of microprocessors

- Communication was performed using bi-directional queues between nearest neighbors

Messages were forwarded by processors on path. ("Store and forward" networking)

There was a strong emphasis on topology in algorithms, in order to minimize the number of hops = minimize time.

To have a large number of different transfers occurring at once, you need a large number of distinct wires (Not just a bus, as in shared memory).

Networks are like streets:

- Link = street

- Switch = intersection.

- Distances (hops) = number of blocks traveled.

- Routing algorithm = travel plan.

Properties:

**Latency: how long to get between nodes in the network.**

- Street: time for one car = dist (miles) / speed (miles/hr)

**Bandwidth: how much data can be moved per unit time.**

- Street: cars/hour = density (cars/mile) * speed (miles/hr) * \# lanes

- Network bandwidth is limited by the bit rate per wire and \# wires

**Topology (how things are connected)**

- Crossbar; ring; 2-D, 3-D, higher-D mesh or torus; hypercube; tree; butterfly; perfect shuffle; dragon fly, ...

**Routing algorithm**

- Example in 2D torus: all east-west then all north-south (avoids deadlock).

**Switching strategy**

- Circuit switching: full path reserved for entire message, like the telephone.

- Packet switching: message broken into separately-routed packets, like the post office, or internet.

**Flow control (what if there is congestion)**

- Stall, store data temporarily in buffers, re-route data to other nodes, tell source node to temporarily halt, discard, etc.

**Diameter the maximum (over all pairs of nodes) of the shortest path between a given pair of nodes.**

**Latency: delay between send and receive times**

- Latency tends to vary widely across architectures

- Vendors often report **hardware latencies** (wire time)

- Application programmers care about **software latencies** (user program to user program)

**Observations:**

- Latencies differ by 1-2 orders across network designs

- Software/hardware overhead at source/destination dominate cost (1s - 10s usecs)

- Hardware latency varies with distance (10s - 100s nsec per hop) but is small compared to overheads

**Latency is key for programs with many small messages.** Latency has not improved significantly, unlike Moore's Law.

The bandwidth of a link = \# wires / time-per-bit

Bandwidth typically in Gigabytes/sec (GB/s), Effective bandwidth is usually lower than physical link bandwidth due to packet overhead. 

**Bandwidth is important for applications with mostly large messages**

Bisection bandwidth: bandwidth across smallest cut that divides network into two equal halves.

Bandwidth across "narrowest" part of the network.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.41y2y8wmu7.webp)

Bisection bandwidth is important for algorithms in which all processors need to communicate with all others.

### Topologies

#### Linear array

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.491atot02e.webp)

- Diameter $=n-1$; average distance $\approx n/3$

- Bisection bandwidth = 1( in units of link bandwidth).

#### Torus or Ring

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.ic58g4q4c.png)

- Diameter $=n/2$; average distance $\approx n/4$

- Bisection bandwidth = 2

- Natural for algorithms that work with 1D arrays.

#### Two dimensional mesh

- Diameter $=2 * (\sqrt{n}-1)$

- Bisection bandwidth = $\sqrt{n}$

#### Two dimensional torus

- Diameter = $\sqrt{n}$

- Bisection bandwidth = $2*\sqrt{n}$

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.2doq12l2rn.webp)

Generalizes to higher dimensions, Natural for algorithms that work with 2D and/or 3D arrays (matmul)

#### Hypercubes

Number of nodes $n=2^d$ for dimension d.

- Diameter = $d$

- Bisection bandwidth = $n/2$

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.5mntxqeqje.webp)

Popular in early machines. (Lots of clever algorithms)

**Greycode addressing**: Each node connected to d others with 1 bit different.

#### Trees

Diameter = $\log n$

Bisection bandwidth = 1.

Easy layout as planar graph, there are many tree algorithms (e.g., summation).

Fat trees avoid bisection bandwidth problem:

- More (or wider) links near top.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.2krxwiituc.webp)

#### Butterflies

- Really an unfolded version of hypercube.

- A d-dimensional butterfly has $(d+1)2^d$ "**switching nodes**" (not to be confused with processors, which is $n=2^d$ )

- Butterfly was invented because hypercube required increasing radix of switches as the network got larger; prohibitive at the time

- Diameter = $\log n$. Bisection bandwidth = n

- No path diversity: bad with adversarial traffic.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.1e8mnx2u7a.webp)

#### Why so many topologies?

- Different systems have different needs (size of the system)

- Complexity vs. optimality

- Physical constraints (Innovations in HW enable previously infeasible technologies)

Two recent technological changes:

- Higher radix (number of ports supported) switches economical, which is really a consequence of Moore's law

- Fiber optic is feasible $\rightarrow$ distance doesn't matter

#### Dragonflies

**Motivation**: Exploit gap in cost and performance between optical interconnects (which go between cabinets in a machine room) and electrical networks (inside cabinet)

- Optical (fiber) more expensive but higher bandwidth when long

- Electrical (copper) networks cheaper, faster when short.

**Combine in hierarchy**:

- Several groups are connected together using all to all links, i.e. each group has at least one link directly to each other group.

- The topology inside each group can be any topology.

**Uses a randomized routing algorithm**

**Outcome**: programmer can (usually) ignore topology, get good performance

- Important in virtualized, dynamic environment.

- Drawback: variable performance.

"Technology-Drive, Highly-Scalable Dragonfly Topology," ISCA 2008

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.1ovgh2y3jv.webp)

Source of image on the right (and more info):

http://www.nersc.gov/users/computational-systems/edison/configuration/interconnect/

### Why randomized routing?

**Minimal routing**: 

- if $Gs \ne Gd$ and $Rs$ has no connection to $Gd$, route within $Gs$ from $Rs$ to $Ra$, which has a global channel to $Gd$

- If $Gs\ne Gd$, traverse the global channel from $Ra$ to router $Rb$ in $Gd$.

- If $Rb\ne Rd$, route from $Rb$ to $Rd$

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.5c104luym9.webp)

Minimal routing works well when things are load balanced, potentially catastrophic in adversarial traffic patterns.

**Randomization idea**: For each packet sourced at router $Rs\in Gs$ and addressed to a router in another group $Rd\in Gd$, first route it to an intermediate group $Gi$.

- this requires at most two group-level link traversals.

- And at most 5 total link traversals.

- Valiant, Leslie G. "A scheme for fast parallel communication." SIAM journal on computing 11.2 (1982): 350-361. 21 2/15/22

### Performance Models

#### Shared Memory Performance Models

Parallel Random Access Memory (PRAM)

All memory access operations complete in one clock period -- no concept of memory hierarchy ("too good to be true")

- OK for understanding whether an algorithm has enough parallelism at all 

- Parallel algorithm design strategy: first do a PRAM algorithm, then worry about memory/communication time (sometimes works)

Slightly more realistic versions exist

- E.g., Concurrent Read Exclusive Write (CREW) PRAM.

- Still missing the memory hierarchy.

#### Latency and Bandwidth Model

Time to send message of length n is roughly
$$\begin{align}
\textbf{Time}&=\textbf{latency}+n*\textbf{cost\_per\_word}\\
&=\textbf{latency} + \textbf{n/bandwidth}
\end{align}$$
Topology is assumed irrelevant.

Often called "$\alpha-\beta$ model" and written 
$$\textbf{Time}=\alpha+n*\beta$$

Usually $\alpha$ >> $\beta$ >> time per flop.

- One long message is cheaper than many short ones.

- Can do hundreds or thousands of flops for cost of one message.
$$
\alpha+n*\beta << n*(\alpha+1*\beta)
$$

Lesson: Need large computation-to-communication ratio to be efficient.

LogP - more detailed model.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.3gofc0323i.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.pfd3xhdo1.webp)

## Programming Distributed Memory Machines using Message Passing

All communication, synchronization require subroutine calls

- No shared variables

- Program run on a single processor just like any uniprocessor program, except for calls to message passing library.

**Communication**

Pairwise or point-to-point: Send and Receive

Collectives all processor get together to 

- Move data: Broadcast, Scatter/gather

- Compute and move: sum, product, max, prefix sum, ... of data on many processors.

**Synchronization**

- Barrier

- No locks because there are no shared variables to protect

**Enquiries**

- How many processes? Which one am I? Any messages waiting?

**Novel Features of MPI**

- **Communicators** encapsulate communication spaces for library safety.

- **Datatypes** reduce copying costs and permit heterogeneity.

- Multiple communication **modes** allow precise buffer management.

- Extensive **collective operations** for scalable global communication.

- **Process topologies** permit efficient process placement, user views of process layout.

- **Profiling interface** encourages portable tools.


MPI Hello world
```c++
#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	MPI_COMM_size( MPI_COMM_WORLD, &size);
	printf("I am %d of %d\n", rank , size);
	MPI_Finalize();
	return 0;
}
```

- All MPI programs begin with `MPI_Init` and end with `MPI_Finalize`.

- `MPI_COMM_WORLD` is defined by `mpi.h` and designates all processes in the MPI "job"

- Each statement executes independently in each process

- The MPI-1 Standard does not specify how to run an MPI program, but many implementations provide
```bash
mpirun -np 4 a.out
```

**Basic Concepts**

- Processes can be collected into groups

- Each message is sent in a context, and must be received in the same context (Provides necessary support for libraries)

- A group and context together form a communicator

- A process is identified by its rank in the group associated with a communicator

- There is a default communicator whose group contains all initial processes, called `MPI_COMM_WORLD`.

**MPI Datatypes**

The data in a message to send or receive is described by a triple (address, count, datatype), where

An MPI datatype is recursively defined as:

- predefined, corresponding to a data type from the language (e.t., MPI_INT, MPI_DOUBLE)

- a contiguous array of MPI datatypes

- a strided block of datatypes

- an indexed array of blocks of datatypes

- an arbitrary structure of datatypes.

There are MPI functions to construct custom datatypes, in particular ones for subarrays.

May hurt performance if datatypes are complex

**MPI Tags**

- Messages are sent with an accompanying user-defined integer tag, to assist the receiving process in identifying the message

- Messages can be screened at the receiving end by specifying a specific tag, or not screened by specifying `MPI_ANY_TAG` as the tag in a receive.

- Some non-MPI message-passing systems have called tags "message types". MPI calls them tags to avoid confusion with datatypes.

```c++
MPI_SEND(start, count, datatype, dest, tag, comm)
```

- the massage buffer is described by (start, count, datatype)

- The target process is specified by `dest`, which is the rank of the target process in the communicator specified by `comm`.

- When this function returns, the data has been delivered to the system and the buffer can be reused. The message may not have been received by the target process.

```c++
MPI_RECV(start, count, datatype, source, tag, comm, status)
```

- Waits until a matching (both `source` and `tag`) message is received from the system, and the buffer can be used.

- `source` is rank in communicator specified by `comm`, or `MPI_ANY_SOURCE`

- `tag` is a tag to be matched or `MPI_ANY_TAG`

- receiving fewer than `count` occurrences of `datatype` us OK, but receiving more is an error 

- `status` contains further information (e.g. size of message)

```c++
// A Simple MPI Program
#include "mpi.h"
#include <stdio.h>
int main(int argc, char* argv[]) {
	int rank, buf;
	MPI_Status status;
	MPI_Init(&argv, &argc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* Process 0 sends and Process 1 receives */
	if (rank == 0) {
		buf = 123456;
		MPI_Send(&buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		MPI_Recv(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		printf("Received %d\n", buf);
	}
	MPI_Finalize();
	return 0;
}
```

- Status is a data structure allocated in the user's program.

```c++
int recvd_tag, recvd_from, recvd_count;
MPI_Status status;
MPI_Recv(..., MPI_ANY_SOURCE, MPI_ANY_TAG, ..., &status)
recvd_tag = status.MPI_TAG;
recvd_from = status.MPI_SOURCE;
MPI_Get_count(&status, datatype, &recvd_count);
```

Claim: most MPI applications can be written with only 6 functions (although which 6 may differ)

Using point-to-point:

- `MPI_INIT`

- `MPI_FINALIZE`

- `MPI_COMM_SIZE`

- `MPI_COMM_RANK`

- `MPI_SEND`

- `MPI_RECEIVE`

Using collectives:

- `MPI_INIT`

- `MPI_FINALIZE`

- `MPI_COMM_SIZE`

- `MPI_COMM_RANK`

- `MPI_BCAST`

- `MPI_REDUCE`

You may use more foe convenience or performance

### Example PI

Mathematically, we know that:
$$
\int_0^1\frac{4.0}{(1+x^2)}dx=\pi
$$
We can approximate the integral as a sum of rectangles:
$$
\overset{N}{\underset{i=0}{\sum}}F(x_i)\delta x\approx\pi
$$
Where each rectangle has width $\delta x$ and height $F(x_i)$ at the middle of interval i.

e.g. in a 4-process run, each process gets every $\text{4}^{\text{th}}$ interval. Process 0 slices are in red. 

Simple program written in a data parallel style in MPI

- E.g. for a reduction (recall "data parallelism" lecture), each process will first reduce (sum) its own values, then call a collective to combine them

Estimates $\pi$ by approximating the area of the quadrant of a unit circle

Each process gets 1/p of the intervals (mapped round robin, a cyclic mapping)

```c++
#include "mpi.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
	int done = 0, n, myid, numprocs, i, rc;
	double PI25DT = 3.141592653589793238462643;
	double mypi, pi, h, sum, x, a;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	while (!done) {
		if (myid == 0) {
			printf("Enter the number of intervals: (0 quits) ");
			scanf("%d", &n);
		}
		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMMWORLD);
		if (n == 0) break;
		h = 1.0 / (double) n;
		sum = 0.0;
		for (i = myid + 1; i <= n; i += numprocs) {
			x = h * ((double)i - 0.5);
			sum += 4.0 * sqrt(1.0 - x * x);
		}
		mypi = h * sum;
		MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (myid == 0) 
			printf("pi is approximately %.16f, Error is .16f\n", pi, fabs(pi - PI25DT));
	}
	MPI_Finalize();
	
	return 0;
}
```
 
So far we have been using *blocking* communication:

- `MPI_Recv` does not complete until the buffer is full (available for use).

- `MPI_Send` does not complete until the buffer is empty (available for use).

Completion depends on size of message and amount of system buffering.

Non-blocking operations return (immediately) "request handles" that can be tested and waited on:

```c++
MPI_Request request:
MPI_Status status;
MPI_Isend(start, count, datatype, dest, tag, comm, &request);
MPI_Irecv(start, count, datatype, dest, tag, comm, &request);
MPI_Wait(&request, &status);
(each request must be Waited on)
```

One can also test without waiting:
```c++
MPI_Test(&request, &flag, &status);
```

Accessing the data buffer without waiting is undefined.

It is sometimes desirable to wait on multiple requests:
```c++
MPI_Waitall(count, array_of_requests, array_of_statuses);
MPI_Waitany(count, array_of_requests, &index, &status);
MPI_Waitsome(count, array_of_requests, array_of_indices, array_of_statuses);
```
There are corresponding versions of test for each of these.

MPI provides multiple *modes* for sending messages:

- Synchronous mode (`MPI_Ssend`): the send does not complete until a matching receive has begun. (Unsafe programs deadlock.)

- Buffered mode (`MPI_Bsend`): the user supplies a buffer to the system for its use. (User allocates enough memory to make an unsafe program safe.)

- Ready mode (`MPI_Rsend`): user guarantees that a matching receive has been posed. Allows access to fast protocols; undefined behavior if matching receive not posted.

Non-blocking versions (`MPI_Issend`)

`MPI_Recv` receives messages sent in any mode.

See www.mpi-forum.org for summary of all flavors of send/receive.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.7zqggbx8y8.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.41y2znmlaq.webp)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4g4iqiv96u.webp)

