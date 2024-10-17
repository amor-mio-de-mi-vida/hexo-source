---
title: Automatic Differentiation
tags:
  - course
categories:
  - DLSystem
date: 2024-10-16 12:52:13
date modified: 2024-10-16 22:44:46
---
## General introduction to different differentiation methods

### Numerical differentiation

$$
\frac{\partial f(\theta)}{\partial \theta_i}=\underset{\epsilon\rightarrow0}{\lim}\frac{f(\theta+\epsilon e_i-f(\theta))}{\epsilon}
$$
$$
\frac{\partial f(\theta)}{\partial\theta_i}=\frac{f(\theta+\epsilon e_i)-f(\theta-\epsilon e_i)}{2\epsilon}+o(\epsilon^2)
$$

### Symbolic differentiation

Write down the formulas, derive the gradient by sum, product and chain rules.

$$
\begin{align}
&\frac{\partial(f(\theta)+g(\theta))}{\partial\theta}=\frac{\partial f(\theta)}{\partial\theta}+\frac{\partial g(\theta)}{\partial\theta} \\
&\frac{\partial(f(\theta)g(\theta))}{\partial \theta}=g(\theta)\frac{\partial f(\theta)}{\partial\theta}+f(\theta)\frac{\partial g(\theta)}{\partial\theta} \\
& \frac{\partial f(g(\theta))}{\partial\theta} = \frac{\partial f(g(\theta))}{\partial g(\theta)}\cdot\frac{\partial g(\theta)}{\partial\theta}
\end{align}
$$

### Computational graph

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.83a147m39b.webp)

## Reverse mode automatic differentiation
$v_1$ is being used in multiple pathways ($v_2$ and $v_3$)
y can be written in the form of $y=f(v_2, v_3)$
$$
\overline{v_1}=\frac{\partial y}{\partial v_1}=\frac{\partial f(v_2, v_3)}{\partial v_2}\cdot\frac{\partial v_2}{\partial v_1}+\frac{\partial f(v_2,v_3)}{\partial v_3}\cdot\frac{\partial v_3}{\partial v_1} = \overline{v_2}\frac{\partial v_2}{\partial v_1}+\overline{v_3}\frac{\partial v_3}{\partial v_1}
$$
Define partial adjoint $\overline{v_{i\rightarrow j}}=\overline{v_j}\frac{\partial v_j}{\partial v_i}$ for each input output node pair $i$ and $j$.
$$
\overline{v_i}=\underset{j\in next(i)}{\sum}\overline{v_{i\rightarrow j}}
$$
we can compute partial adjoints separately then sum them together.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.2doosyqnsw.webp)

### Reverse mode AD on tensors

Forward evaluation trace

$$\begin{align}
&Z_{ij}=\underset{k}{\sum}X_{ik}W_{kj}\\
&v=f(Z)
\end{align}$$

Forward matrix form

$$\begin{align}
&Z=XW\\
&v=f(Z)
\end{align}$$

**Define adjoint** for tensor values 
$$
\overline{Z}=
\left[
\begin{matrix}
&\frac{\partial y}{\partial Z_{1,1}} & \cdots & \frac{\partial y}{\partial Z_{1,n}} \\
&\cdots & \cdots & \cdots \\
&\frac{\partial y}{\partial Z_{m,1}}&\cdots&\frac{\partial y}{\partial Z_{m,n}}
\end{matrix}
\right]
$$

Reverse evaluation in scalar form

$$
\overline{X_{i,k}}=\underset{j}{\sum}\frac{\partial Z_{i,j}}{\partial X_{i,k}}\overline{Z_{i,j}}=\underset{j}{\sum}W_{k,j}\overline{Z_{i,j}}
$$

Reverse matrix form

$$
\overline{X}=\overline{Z}W^T
$$

### Differentiable Programming


