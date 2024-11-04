---
date: 2024-11-03 12:15:36
date modified: 2024-11-03 20:11:44
title: Model Deployment
tags:
  - course
categories:
  - DLSystem
---
## Model deployment overview

### Model deployment considerations

Application environment bring restrictions (model size, no-python)

Leverage local hardware acceleration (mobile GPUs, accelerated CPU instructions, NPUs)

Integration with the applications (data preprocessing, post processing)

### Inference engine internals

Many inference engines are structured as computational graph interpreters.

Allocate memories for intermediate activations, traverse the graph and execute each of the operators

Usually only support a limited set of operators and programming models.(e.g. dynamism)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.7ax6c5i3lq.webp)

## Machine learning compilation

### Limitation of Library driven inference engine deployments

Need to build specialized libraries for each hardware backend

A lot of engineering efforts to optimization.

ML Models $\Rightarrow$ High-level IR Optimizations and Transformations $\Rightarrow$ Tensor Operator Level Optimization $\Rightarrow$ Direct code generation.

### High-level IR and optimizations

Computation graph (or graph-like) representation

Each node is a tensor operator (e.g. convolution)

Can be transformed (e.g. fusion) and annotated (e.g. device placement)

Most ML frameworks have this layer.

### Search via learned cost model

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.4n7q1tmz1s.webp)

### Summary: elements of an automated ML compiler

Program representation

- Represent the program/optimization of interest, (e.g. dense tensor linear algebra, data structures)

Build search space through a set of transformations

- Cover common optimizations

- Find ways for domain experts to provide input

Effective search

- Cost models, transferability

