---
date: 2024-10-18 10:24:02
date modified: 2024-10-18 13:31:06
title: Neural Network Library Abstractions
tags:
  - course
categories:
  - DLSystem
---
## Programming abstractions

### Forward and backward layer interface

Defines the forward computation and backward (gradient) operation.

Used in cuda-convenet (the AlexNet framework)

### Computational graph and declarative programming

```python
import tensorflow as tf

v1 = tf.Variable()
v2 = tf.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

sess = tf.Session()
value4 = sess.run(v4, feed_dict={v1: numpy.array([1])})

```

First declare the computational graph, 

Then execute the graph by feeding input value.

- It gives you an opportunity to look at a computational graph and skip some computation that are unnecessary.

- Having a complete computational graph gives you a lot of opportunities for optimization

- The place where you run the computational graph does not have to be on the same local machine.

### Imperative automatic differentiation

```python 
import neelde as ndl

v1 = ndl.Tensor([1])
v2 = ndl.exp(v1)
v3 = v2 + 1
v4 = v2 * v3
```

Executes computation as we construct the computational graph, Allow easy mixing of python control flow and construction.

```python
if v3.numpy() > 0.5:
	v5 = v4 * 2
else:
	v5 = v4
```

- the flexibility of the imperative automatic differentiation that allows you to organically construct these dynamic computational graphs really gives the researchers much more ability to be able to express ideas.

What are the pros and cons of each programming abstraction?

## High level modular library components

Keys to consider:

 - for given inputs, how to compute outputs

- get the list of (trainable) parameters

- ways to initialize the parameters

- how to compose multiple objective functions together?

- what happens during inference time after training?

### Regularization and optimizer

Two ways to incorporate regularization:

- Implement as part of loss function

- Directly incorporate as part of optimizer update

### Initialization

### Data loader and preprocessing

Two levels of abstractions

- Computational graph abstraction on Tensors, handles AD.

- High level abstraction to handle modular composition.