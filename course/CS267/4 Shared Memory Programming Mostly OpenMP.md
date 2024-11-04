---
date: 2024-11-04 19:20:29
date modified: 2024-11-04 19:57:01
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

