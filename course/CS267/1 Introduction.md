---
date: 2024-11-02 09:42:50
date modified: 2024-11-02 10:53:29
title: Introduction
tags:
  - course
categories:
  - cs267
---
## What is a Parallel Computer ?

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/CS-267/image.4ckw71lo0u.webp)

**SMP**

A shared Memory multiprocessor (SMP\*) by connecting multiple processors to a single memory system.

A multicore processor contains multiple processors (cores on a single chip)

**HPC**

A distributed memory multiprocessor has processors with their own memories connected by a high speed network.

Also called a cluster

**SIMD**

A Single Processor Multiple Data (SIMD) computer has multiple processors (or functional units) that perform the same operation on multiple data elements at once

Most single processors have SIMD units with 2~8 way parallelism

Graphics processing units (GPUs) use this as well.

> Performance = parallelsim
> Efficiency = locality

## How to cover all applications?

| 7 Giants of Data   | 7 Dwarfs of Simulation |
| ------------------ | ---------------------- |
| Basic statistics   | Monte Carlo methods    |
| Generalized N-Body | Particle methods       |
| Graph-theory       | Unstructured meshes    |
| Linear algebra     | Dense Linear Algebra   |
| Optimizations      | Sparse Linear Algebra  |
| Integrations       | Spectral methods       |
| Alignment          | Structured Meshes      |

## Overview of the course (not in order)

Parallel Programming Models and Machines (plus some architecture, e.g., caches)

| Algorithm/machine model | Language/Library skills |
| ----------------------- | ----------------------- |
| Shared memory           | OpenMP (pthreads)       |
| Distributed memory      | MPI                     |
|                         | PGAS                    |
| Data Parallel           | SPARK                   |
|                         | CUDA                    |

Performance models

- Roofline

- $\alpha-\beta$ (latency/bandwidth)

- (LogP)

Cross-cutting topics:

- Communication avoiding

- Load balancing

- Hierarchical algorithms

- Autotuning

The Laws

- Moore's Law

- Amdahl's Law

- Little's Law

Applications (in some detail)

- Machine Learning

- Biology

- Cosmology