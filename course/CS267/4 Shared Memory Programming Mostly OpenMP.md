---
date: 2024-11-04 19:20:29
date modified: 2024-11-08 11:39:23
title: Shared Memory Programming Mostly OpenMP
tags:
  - course
categories:
  - cs267
---
## Shared Memory

- Program is a collection of threads of control (Can be created dynamically, mid-execution, in some languages).

- Each thread has a set of private variables, e.g., local stack variables

- Also a set of shared variables, e.g., static variables, shared common blocks, or global heap.

Threads communicate **implicitly** by writing and reading shared variables.

Threads coordinate by **synchronizing** on shared variables.

## Parallel Programming with Threads

```c++
// Signature:
int pthread_create(pthread_t *, const pthread_attr_t*,
				    void* (*)(void*),
				    void*);

// Example call
errcode = pthread_create(&thread_id, &thread_attribute, 
						&thread_fun, &fun_arg);
```
`thread_id` is the thread id or handle (used to halt, etc.)

`thread_attribute` various attributes

- Standard default values obtained by passing a NULL pointer.

- Sample attributes: minimum stack size, priority.

`thread_fun` the function to be run (takes and returns void\*)

`fun_arg` an argument can be passed to thread_fun when it starts

`errorcode` will be nonzero if the create operation fails.

### Recall Data Race Example

Problem is a race condition on variable s in the program

A race condition or data race occurs when:

- two processors (or two threads) access the same variable, and at least one does a write.

- The accesses are concurrent (not synchronized) so they could happen simultaneously.

Mutexes —— mutual exclusion aka locks

- threads are working mostly independently

- need to access common data structure

```c++
lock* mutex = alloc_and_init(); /* shared */
acquire(1);
	access data
release(1);
```

- Locks only affect processors using them: if a thread accesses the data without doing the acquire/release, locks by others will not help

- Java, C++ and other languages have lexically scoped synchronization, i.e., synchronized methods/blocks

- Semaphores (a signaling mechanism) generalize locks to allow k threads simultaneous access; good for limited resources.

- Unlike in a mutex, a semaphore can be decremented by another process (a mutex can only be unlocked by its owner)

```c++
// To create a mutex:
#include <pthread.h>
pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;
// or pthread_mutex_init(&amutex, NULL)

// To use it:
int pthread_mutex_lock(amutex);
int pthread_mutex_unlock(amutex);

// To deallocate a mutex
int pthread_mutex_destroy(pthread_mutex_t * mutex)
```

- Multiple mutexes may be held, but can lead to problems.

- Dead lock results if both threads acquire one of their locks, so that neither can acquire the second.

**Pitfalls**

- Overhead of thread creation is high (1-loop iteration probably too much)

- Data race bugs are very nasty to find because they can be intermittent

Researchers look at transactional memory an alternative

OpenMP is commonly used today as an alternative

- Helps with some of these, but doesn't make them disappear.

As application programmers, you're better off avoiding these P-thread programming due to the pitfalls such as the need to manually assign work, balance it out, and make sure there's no oversubscription, make sure the deadlocks do not happen.

### A Programmer's View of OpenMP

OpenMP is a portable, threaded, shared-memory programming *specification* with "light" syntax.

- Requires compiler support

OpenMP will:

- Allow a programmer to separate a program into *serial regions* and *parallel regions*, rather than P concurrently-executing threads.

- Hide stack management

- Provide synchronization constructs.

OpenMP will not:

- Parallelize automatically

- Guarantee speedup

- Provide freedom from data races


| OpenMP pragma, function, or clause                          | Concepts                                                                                                                           |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `#pragma omp parallel`                                      | Parallel region, teams of threads, structured block, interleaved execution across threads                                          |
| `int omp_get_thread_num()`<br>`int omp_get_num_threads()`   | Create threads with a parallel region and split up the work using the number of threads and thread ID                              |
| `double omp_get_wtime()`                                    | Speedup and Amdahl's law.<br>False Sharing and other performance issues                                                            |
| `setenv OMP_NUM_THREADS N`                                  | Internal control variables. Setting the default number of threads with an environment variable                                     |
| `#pragma omp barrier`<br>`#pragma omp critical`             | Synchronization and race conditions. Revisit interleaved execution.                                                                |
| `#pragma omp for`<br>`#pragma omp parallel for`             | Worksharing, parallel loops, loop carried dependencies                                                                             |
| `reduction(op:list)`                                        | Reductions of values across a team of threads                                                                                      |
| `schedule(dynamic [,chunk])`<br>`schedule(static [,chunk])` | Loop schedules, loop overheads and load balance                                                                                    |
| `private(list)`<br>`firstprivate(list)`<br>`shared(list)`   | Data environment                                                                                                                   |
| `nowait`                                                    | Disabling implied barriers on workshare constructs, the high cost of barriers, and the flush concept (but not the flush directive) |
| `#pragma omp single`                                        | Workshare with a single thread                                                                                                     |
| `#pragma omp task`<br>`#pragma omp taskwait`                | Tasks including the data environment for tasks.                                                                                    |

Switches for compiling and linking 

`gcc -fopenmp` $\Rightarrow$ GNU (Linux, OSX)

`pgcc -mp pgi` $\Rightarrow$ PGI (Linux)

`icl/Qopenmp` $\Rightarrow$ Intel (windows)

`icc -fopenmp` $\Rightarrow$ Intel (Linux, OSX)

OpenMP is not forced to kill the threads and recreate them. It can suspend the threads and send them back to a thread pool and reuse them in the next parallel region generation.

You can request a number of threads with `omp_set_num_threads()`

But is the number of threads requested the number you actually get?

- NO! An implementation can silently decide to give you a team with fewer threads

- Once a team of threads is established ... the system will not reduce the size of the team. this is only essentially a suggestion to the OpenMP runtime.

## Shared Memory Hardware and Memory Consistency

Want high performance for shared memory: Use Caches!

- Each processor has its own cache (or multiple caches)

- Place data from memory into cache

- Writeback cache: don't send all writes over bus to memory

Caches reduce average latency

- Automatic replication closer to processor

- *More* important to multiprocessor than uniprocessor: latencies longer

Normal uniprocessor mechanisms to access data

- Loads and Stores form very low-overhead communication primitive 

**Problem: Cache Coherence!**

### Cache Coherence Problem

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.ic4ro6tcr.webp)

Things to note:

- Processors could see different values for u after event 3

- With write back caches, value written back to memory depends on happenstance of which cache flushes or writes back value when

How to fix with a bus: Coherence Protocol

- Use bus to broadcast writes or invalidations

- Simple protocols rely on presence of broadcast medium

Bus not scalable beyond about 100 processors

- Capacity, bandwidth limitations.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.26lhov4w8w.webp)

### Parallel PI Program

```c++
#include <omp.h>
static long num_steps = 100000; double step;
#define NUM_THREADS 4

void main() {
	int i, nthreads; 
	double pi, sum[NUM_THREADS];
	step = 1.0/(double)num_steps;
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, nthrds;
		double x;
		id = omp_get_thread_num();
		nthrds = omp_get_num_threads();
		if (id == 0) nthreads = nthrds;
		for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds) {
			x = (i + 0.5) * step;
			sum[id] += 4.0/(1.0 + x * x);
		}
	}	
	for (i=0, pi=0.0; i < nthreads; i++)
		pi += sum[i] * step;
}
```
For each threads, we're getting like the every nthreads variable.

### Why such poor scaling? False sharing

if independent data elements happen to sit on the same cache line, each update will cause the cache lines to "slosh back and forth" between threads ... This is called "false sharing"

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.8hghnhff5f.webp)

- If you promote scalars to an array to support creation of an SPMD program, the array elements are contiguous in memory and hence share cache lines ... Results in poor scalability

- Solution: Pad arrays so elements you use are on distinct cache lines.

Eliminate false sharing by padding the sum array

```c++
#include <omp.h>
static long num_steps = 100000; 
double step;
#define PAD 8 // assume 64 byte L1 cache line size
#define NUM_THREADS 4

void main() {
	int i, nthreads; 
	double pi, sum[NUM_THREADS][PAD];
	// Pad the array so each sum value is in a different cache line
	step = 1.0/(double)num_steps;
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, nthrds;
		double x;
		id = omp_get_thread_num();
		nthrds = omp_get_num_threads();
		if (id == 0) nthreads = nthrds;
		for (i = id, sum[id] = 0.0; i < num_steps; i = i + nthrds) {
			x = (i + 0.5) * step;
			sum[id][0] += 4.0/(1.0 + x * x);
		}
	}	
	for (i=0, pi=0.0; i < nthreads; i++)
		pi += sum[i][0] * step;
}

```

### Synchronization

Synchronization is used to impose order constraints and to protect access to shared data.

**Synchronization: critical**

Mutual exclusion: Only one thread at a time can enter a critical region.

```c++
float res;
#pragma omp parallel
{
	float B; 
	int i, id, nthrds;
	id = omp_get_thread_num();
	nthrds = omp_get_num_threads();
	for (i=id; i<niters; i+=nthrds) {
		B=big_jon(i);
		#pragma omp critical
		// Threads wait their turn -- only one at a time calls consume() 
		res+=consume(B);
	}

}
```

**Synchronization: barrier**

Barrier: a point in a program all threads much reach before any threads are allowed to proceed.

It is a "stand alone" pragma meaning it is not associated with user code ... it is an executable statement.

```c++
double Arr[8], Brr[8]; 
int numthrds;
omp_set_num_threads(8);
#pragma omp parallel
{
	int id, nthrds;
	nthrds = omp_get_num_threads();
	if (id == 0) numthrds = nthrds;
	Arr[id] = big_ugly_calc(id, nthrds);
	#pragma omp barrier
	// Threads wait until all threads hit the barrier. Then they can go on.
	Brr[id] = really_big_and_ugly(id, nthrds, Arr);
}
```

Using a critical section to remove impact of false sharing

```c++
#include <omp.h>
static long num_steps = 100000; 
double step;
#define NUM_THREADS 4

void main() {
	int nthreads;
	double pi=0.0;
	step = 1.0/(double)num_steps;
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, nthrds;
		double x, sum;
		// Create a scalar local to each thread to accumulate partial sums
		id = omp_get_thread_num();
		nthrds = omp_get_num_threads();
		if (id == 0) nthreads = nthrds;
		for (i = id, sum = 0.0; i < num_steps; i = i + nthrds) {
			x = (i + 0.5) * step;
			sum += 4.0/(1.0+x*x);
			// No array, so no false sharing
		}
		#pragma omp critical
		pi += sum * step;
		// Sum goes "out of scope" beyond the parallel region ... so you must sum it in here. Must protect summation into pi in a critical region so updates don't conflict.
	}	
}
```

How can we not do that and let the compiler do the work sharing?

- The loop work sharing construct splits up loop iterations among the threads in a team

```c++
#pragma omp parallel
{
#pragma omp for
	for (I=0;I<N;I++) {
	// The loop control index I is made "private" to each thread by default
		NEAT_STUFF(I);
	}
	// Threads wait here until all threads are finished with the parallel loop before any proceed past the end of the loop
}
```

In general, can we safely assume that two independent declared variables in the global heap will not reside in the same cache line? —— It's hard to say. Compiler wouldn't give a whole cache line of space between consecutive variables on the heap declared independently from each other.

### Loop worksharing constructs: The schedule clause

The schedule clause affects how loop iterations are mapped onto threads

`schedule (static [, chunk])` Deal-out blocks of iteration of size "chunk" to each thread.

`schedule (dynamic [, chunk])` Each thread grabs "chunk" iterations off a queue until all iterations have been handled.

| Schedule Clause | When To use                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| STATIC          | Pre-determined and predictable by the programmer<br>Least work at runtime: scheduling done at compile-time           |
| DYNAMIC         | Unpredictable, highly variable work per iteration<br>Most work at runtime: complex scheduling logic used at run-time |

You need to somehow decompose actual problem into smaller pieces if you have variable work in each iteration so that the dynamic can help you in runtime. 

If you have few highly variable workloads, chunks, then there's not much any scheduling can do for you.

### Static vs dynamic example

- Suppose you are doing **block matrix multiplication**, i.e. you split each matrix into $B^2$ submatrices, each of size $N/B$ by $N/B$ and will do $B^3$ submatrix multiplications (B is relatively small, say ~10)

- However, matrices are sparse and submatrices have varying number of nonzeros.

```c++
#pragma omp parallel for collapse(2)
for (int i = 0; i < B; i++) {
	for (int j = 0; j < B; j++) {
// collapsing the first two loops
		for (int k = 0; k < B; k++) {
			SpGEMM(A(i,k), B(k,j), C(i,j));
		}
	}
}
```

these are sparse matrix multiplications and they have variable workload. The default scheduling is static and it's likely dividing this up in a load imbalance way. We can use the schedule dynamic.

```c++
#pragma omp parallel schedule(dynamic) for collapse(2)
for (int i = 0; i < B; i++) {
	for (int j = 0; j < B; j++) {
// collapsing the first two loops
		for (int k = 0; k < B; k++) {
			SpGEMM(A(i,k), B(k,j), C(i,j));
		}
	}
}
```

If you have difficulty parallelizing certain loops, because there's a loop dependency. **The basic approach to fix this thing is to change that variable to eliminate the loop dependency**

Basic approach 

- Find compute intensive loops

- Make the loop iterations independent ... So they can safely execute in any order without loop-carried dependencies

- Place the appropriate OpenMP directive and test.

### Reduction

OpenMP reduction clause `reduction(op : list)`

Inside a parallel or a work-sharing construct:

- A local copy of each list variable is made and initialized depending on the "op" (e.g. 0 for "+")

+ Updates occur on the local copy.

- Local copies are reduced into a single value and combined with the original global value.

The variables in "list" must be shared in the enclosing parallel region.

```c++
double ave=0.0, A[MAX]; int i;
#pragma omp parallel for reduction (+:ave)
for (i = 0; i < MAX;i++) {
	ave += A[i];
}
ave = ave/MAX;
```

```c++
#include <omp.h>
static long_num_steps = 100000;
double step;
void main() {
	int i;
	double x, pi, sum = 0.0;
	step = 1.0/(double)num_steps;
	#pragma omp parallel
	// create a team of threads ... without a parallel construct, you'll never have more than one thread
	{
		double x;
		// create a scalar local to each thread to hold value of x at the center of each interval
		#pragma omp for reduction(+:sum)
		for (i = 0; i < num_steps; i++) {
		// break up loop iterations and assign them to threads ... setting up a reduction into sum. Note ... the loop index is local to a thread by default.
			x = (i+0.5)*step;
			sum = sum + 4.0/(1.0 + x*x);
		}
		pi = step * sum;
	}

}
```

### The nowait clause

Barriers are really expensive. You need to understand when they are implied and how to skip them when its safe to do so.

```c++
double A[big], B[big], C[big];
#pragma omp parallel
{
	int id = omp_get_thread_num();
	A[id] = big_calc1(id);
	#pragma omp barrier
	#pragma omp for
		for(i=0; i < N; i++) {
			C[i]=big_calc3(i, A);
		}
		//implicit barrier at the end of a for worksharing construct.
	#pragma omp for nowait
		for (i = 0; i < N; i++) {
			B[i] = big_calc2(C, i);
		}
		// no implicit barrier due to nowait
		A[id] = big_calc4(id);
}
// implicit barrier at the end of a parallel region
```


### Data environment: Default storage attributes

Shared memory programming model:

- Most variables are shared by default

Global variables are SHARED among threads

- Fortran: COMMON blocks, SAVE variables, MODULE variables

- C: File scope variables, static

- Both: dynamically allocated memory (ALLOCATE, malloc, new)

But not everything is shared ...

- Stack variables in subprograms (Fortran) or functions (C) called from parallel regions are PRIVATE

- Automatic variables within a statement block are PRIVATE.

### Data sharing: Changing storage attributes

One can selectively change storage attributes for constructs using the following clauses

- `shared(list)`

- `private(list)`

- `firstprivate(list)`

These clauses apply to the OpenMP construct NOT to the entire region.

These can be used on parallel and for constructs ... other than shared which can only be used on a parallel construct

**Force** the programmer to explicitly define storage attributes

#### Private clause

`private(var)` creates a new local copy of var for each thread

- The value of the private copies is **uninitialized**

- The value of the original variable is unchanged after the region

```c++
void wrong() {
	int tmp = 0;
	#pragma omp parallel for private(tmp)
	for (int j = 0; j < 1000; j++) 
		tmp += j; // <- tmp was not initialized
	printf("%d\n",tmp); // <- tmp is 0 here
}
```

When you need to reference the variable tmp that exists prior to the construct, we call it the **original variable**.

#### First private clause

Variables initialized from a shared variable

C++ objects are copy-constructed

```c++
incr = 0;
#pragma omp parallel for firstprivate(incr)
for (i = 0; i <= MAX; i++) {
	if ((i%2)==0) incr++;
	A[i] = incr; // <- Each thread gets its own copy of incr with an initial value of 0
}
```

### Data sharing: Default clause

`default(none)`: Forces you to define the storage attributes for variables that appear inside the static extent of the construct ... if you fail the complier will complain. Good programming practice!

You can put the default clause on parallel and parallel + workshare constructs.

```c++
#include <omp.h>
int main() {
	int i, j=5;
	double x = 1.0, y=42.0;
	#pragma omp parallel for default(none) reduction(*:x)
	for (i = 0; i < N; i++) {
		for (j = 0; j < 3; j++) {
			x += foobar(i, j, y);
		}
	} // <- The static extent is the code in the compliation unit that contains the construct.
	 printf("x is %f\n", (float)x);
}
```
The compiler would complain about j and y, which is important since you don'w want j to be shared.

The full OpenMP specification has other versions of the default clause, but they are not used very often so we skip them in the common core.

### What are tasks?

Tasks are indenpendent units of work

Tasks are composed of:

- code to execute

- data to compute with

Threads are assigned to perform the work of each task

- The thread that encounters the task construct may execute the task immediately

- The threads may defer execution until later

The task construct includes a structured block of code

Inside a parallel region, a thread encountering a task construct will package up the code block and its data for execution

Tasks can be nested: i.e. a task may itself generate tasks.

A common Pattern is to have one thread create the tasks while the other threads wait at a barrier and execute the tasks.

### Single worksharing Construct

- The single construct denotes a block of code that is executed by only one thread (not necessarily the master thread).

- A barrier is implied at the end of the single block (can remove the barrier with a *nowait* clause)

```c++
#pragma omp parallel 
{
	do_many_things();
	#pragma omp single
	{ exchange_boundaries(); }
	do_many_other_things();
}
```

Task Directive

```c++
#pragma omp parallel 
{
	#pragma omp single
	{
		#pragma omp task
		fred();
		#pragma omp task
		daisy();
		#pragma omp task
		billy();
	} // <- All tasks complelte before this barrier is released
}
```

You can't start creating tasks within a parallel region. You have to first make sure that the control goes back to a single thread that will fire up all the tasks and will get distributed.

The OMP parallel task is one of the few constructs in OpenMP that doesn't have an implicit weight after it.

### When/where are tasks complete?

At thread barriers (explicit or implicit)

- applies to all tasks generated in the current parallel region up to the barrier

At taskwait directive

- i.e. wait until all tasks defined within the scope of the current task have completed.

- Note: applies only to tasks generated in the current task, not to "descendants".

- To also wait for descendants, there is the taskgroup *region*

```c++
#pragma omp parallel
{
	#pragma omp single
	{
		#pragma omp task
			fred();
		#pragma omp task
			daisy();
		#pragma taskwait
		#pragma omp task 
			billy(); 
	}
}

// fred() and daisy() must complete before billy() starts
```

The behavior you want for tasks is usually firstprivate, because the task may not be executed until later (and variables may have gone out of scope)

- Variables that are private when the task construct is encountered are firstprivate by default 

Variables that are shared in all constructs starting from the innermost enclosing parallel construct are shared by default

```c++
#pragma omp parallel shared(A) private(B)
{
	...
	#pragma omp task
	{
		int C;
		compute(A, B, C); 
		// A is shared B is firstprivate C is private
	}
}
```

### Fibonacci numbers

```c++
int fib(int n) {
	int x, y;
	if (n < 2) return n;
	
	x = fib(n-1);
	y = fib(n-2);
	return (x+y);
}

int main() {
	int NW = 5000;
	fib(NW);
}
```

```c++
int fib(int n) {
	int x, y;
	if (n < 2) return n;
	
	#pragma omp task shared(x)
	x = fib(n-1);
	#pragma omp task shared(y)
	y = fib(n-2);
	#pragma omp taskwait
	return x+y;
}

int main() {
	int NW = 5000;
	#pragma omp parallel
	{
		#pragma omp single
		fib(NW);
	}
}
```

- Binary tree of tasks

- Traversed using a recursive function

- A task cannot complete until all tasks below it in the tree are complete (enforced with taskwait)

- By default, only 2 threads will be activate in most implementations. Set `OMP_MAX_ACTIVE_LEVELS` with n > 1 to get n-levels of **nested parallelism**

- x, y are local, and so by default they are private to current task. **must be shared on child tasks** so they don't create their own firstprivate copies at this level!

### Synchronization: atomic

Atomic provides mutual exclusion but only applies to the update of a memory location (the update of X in the following example)

```c++
#pragma omp parallel
{
	double B;
	B = DOIT();
	
	#pragma omp atomic
	X += big_ugly(B);
	// Atomic only protects the read/update of X
}
```

### Flush operation

Defines a sequence point at which a thread enforces a consistent view of memory

For variables visible to other threads and associated with the flush operation (the **flush-set**)

- The compiler can't move loads/stores of the flush-set around a flush:
	- All previous read/writes of the flush-set by this thread have completed
	- No subsequent read/writes of the flush-set by this thread have occurred

- Variables in the flush set are moved from temporary storage to shared memory.

- Reads of variables in the flush set following the flush are loaded from shared memory.

<mark>The flush makes the calling threads temporary view match the view in shared memory. Flush by itself does not force synchronization</mark>

Flush forces data to be updated in memory so other threads see the most recent value

```c++
double A;
A = compute();
#pragma omp flush(A)
// flush to memory to make sure other threads can pick up the right value
```

Flush without a list: flush set is all thread visible variables

Flush with a list: flush set is the list of variables

> OpenMP's flush is analogous to a fence in other shared memory APIs

A flush operation is implied by OpenMP synchronizations,

- at entry/exit of parallel regions

- at implicit and explicit barriers

- at entry/exit of critical regions

- whenever a lock is set or unset

(but do not at entry to worksharing regions or entry/exist of master regions)


## Take Away

**Programming shared memory machines**

- May allocate data in large shared region without too many worries about where

- Memory hierarchy is critical to performance. Even more so than on uniprocessors, due to coherence traffic

- For performance tuning, watch sharing (both true and false)

**Semantics**

- Need to lock access to shared variable for read-modify-write.

- Sequential consistency is the natural semantics. Write race-free programs to get this.

- Architects worked hard to make this work. Caches are coherent with buses or directories; No caching of remote data on shared address space machines.

- But compiler and processor may still get in the way. Non-blocking writes, read prefetching, code motion ...; Avoid races or use machine-specific fences carefully.