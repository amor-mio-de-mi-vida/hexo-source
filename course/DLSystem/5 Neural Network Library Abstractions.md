---
date: 2024-10-18 10:24:02
date modified: 2024-10-19 14:54:54
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


The problem is that we are actively  building up a computational graph that tracks the history of all the past updates. Such un-necessary graph tracking can cause memory and speed issues.

Instead, we can create a "detached" tensor that does not requires grad.

### Numerical Stability

most computations in deep learning model are executed using 32-bit floating point. We need to pay special attention to potential numerical problems. Softmax is one of the most commonly used operators in loss functions. Let $z=softmax(x)$, then we have 
$$
z_i=\frac{exp(x_i)}{\sum_kexp(x_k)}
$$
passing a large number (that is greater than 0) to exp function can easily result in overflow. Note that the following invariance hold for any constant $c$.

$$
z_i=\frac{exp(x_i)}{\sum_kexp(x_k)}=\frac{exp(x_i-c)}{\sum_kexp(x_k-c)}
$$

we can pick $c=max(x)$ so that all the inputs to the exp become smaller or equal to 0.

### Designing a Neural Network Library

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.2krwsepgn5.webp)

### Initialization

Under a linear relu network where $y^{(l)}=x^{(l-1)}W^T$, $x^{(l)}=max(y^{(l)},0)$. Assume that $W\in \mathbb{R}^{n_{out}\times n_{in}}$. A common way to do so is to initialize it as $\mathcal{N}(0,\sigma^2)$ where $\sigma=\sqrt{\frac{2}{n_{}in}}$.

Checkout Explanation from the original paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

$$\begin{align}
&y_i=\overset{n_{in}}{\underset{j=1}{\sum}}x_jW_{i,j}\\
&\mathbf{Var}[y_i]=n_{in}\mathbf{E}[x_0^2]\mathbf{Var}[W_{i,j}]=n_{in}\mathbf{E}[x_0^2]\sigma^2 
\end{align}$$

Considering the fact that x is also a result of relu of previous layer

$$
\mathbf{E}[x_0^2]=\mathbf{E}[relu(y^{(l-1)})^2]=\frac{1}{2}\mathbf{Var}[y^{(l-1)}]
$$

we can get the variance value by requiring $\mathbf{Var}[y^{(l-1)}]=\mathbf{Var}[y^{(l)}]$. NOTE: the variance value was derived under a specific deep relu network.


