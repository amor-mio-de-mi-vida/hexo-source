---
date: 2024-10-29 15:06:13
date modified: 2024-10-29 15:39:55
title: Welcome to use Hexo Theme Keep
tags:
  - Hexo
  - Keep
categories:
  - Hexo
---
## Question 1 : Implementing forward computation

### MatMul

Notice that (we use `@` to represent matrix multiplication) for $A\in\mathbb{R}^{m\times n}$, $B\in\mathbb{R}^{n\times k}$, $A@B\in\mathbb{R}^{m\times k}$, the shape of the grad of A should be $\mathbb{R}^{m\times n}$, because each element should have a gradient. 

It can be proof that 

$$\begin{align}
&\frac{\partial (A @ B)}{\partial A} = \text{grad}(A@B) @ B^T = B^T\\
&\frac{\partial (A @ B)}{\partial B} = A^T@\text{grad}(A@B) = A^T
\end{align}$$


## Summation

we can view summation as matrix multiplication for example: 

$$\begin{align}
A = \left[
\begin{matrix}
a_{00} & a_{01} &\cdots& a_{0n} \\
a_{10} & a_{11} &\cdots& a_{1n} \\
\cdots & \cdots &\cdots& \cdots \\
a_{m0} & a_{m1} &\cdots& a_{mn} \\
\end{matrix}
\right]
\overset{\text{sum, axis=1}}{\rightarrow}
\left[
\begin{matrix}
\overset{n}{\underset{i_0=1}{\sum}}a_{0i_0}\\
\overset{n}{\underset{i_1=1}{\sum}}a_{1i_1}\\
\cdots\\
\overset{n}{\underset{i_m=1}{\sum}}a_{mi_m}\\
\end{matrix}
\right]
\end{align}$$

equals to 

$$
\left[\begin{matrix}
a_{00} & a_{01} &\cdots& a_{0n} \\
a_{10} & a_{11} &\cdots& a_{1n} \\
\vdots & \vdots &\cdots& \vdots \\
a_{m0} & a_{m1} &\cdots& a_{mn} \\
\end{matrix}\right]
@ 
\left[\begin{matrix}
1\\
1\\
\vdots\\
1
\end{matrix}\right]
=

$$