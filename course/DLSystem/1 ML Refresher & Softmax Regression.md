---
title: ML Refresher & Softmax Regression
tags:
  - course
categories:
  - DLSystem
date: 2024-10-06 15:50:07
date modified: 2024-10-21 09:15:59
---
## Three ingredients of a machine learning algorithm
Every machine leaning algorithm consists of three different elements
- The hypothesis class: the "program structure", parameterized via a set of *parameters*, that describes how we map inputs to outputs.
- The loss function: a function that specifies how "well" a given hypothesis performs on the task of interest.
- An optimization method: a procedure for determining a set of parameters that (approximately) minimize the sum of losses over the training set.
## Softmax regression
### Linear hypothesis function
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006172352.45hnfj9e3d.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006162754.51e4uzj2j4.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006160438.9kg5xyojgr.webp)
### Loss function
simple version:
$$l_{err}(h(x), y)=
\begin{cases}
0& \text{if argmax$_ih_i(x)=y$} \\
1& \text{otherwise}
\end{cases}$$
this loss function is very bad for optimization (selecting the best parameters). It doesn't provide any information, not differentiable.
softmax version:
$$z_i=p(\text{label}=i)=\frac{exp(h_i(x))}{\underset{j=1}{\overset{k}{\sum}}exp(h_j(x))} \iff z\equiv \text{normalize}(exp(h(x))) $$
$$l_{ce}(h(x), y)=-\log p(\text{label}=y)=-h_y(x)+\log \sum_{j=1}^kexp(h_j(x))$$
### The softmax regression optimization problem
$$f(\theta) = \underset{\theta}{minimize} \frac{1}{m}\underset{i=1}{\overset{m}{\sum}}l(h_{\theta}(x^{(i)}), y^{(i)}) = \underset{\theta}{minimize} \frac{1}{m}\underset{i=1}{\overset{m}{\sum}}l(\theta^T x^{(i)}, y^{(i)})$$
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006163715.9nzrvohm6r.webp)
To minimize a function, the gradient descent algorithm proceeds by iteratively taking steps in the direction of the negative gradient
$$\theta:=\theta-\alpha \nabla_{\theta}f(\theta) $$
where $\alpha >0$ is a step or *learning rate*.
So, how do we compute the gradient for the softmax objective?
$$\nabla_{\theta}l_{ce}(\theta^Tx, y)=\text{?}$$
for vector $h \in \mathbb{R}^k$
$$\begin{align}
\frac{\partial l_{ce}(h, y)}{\partial h_i} &= \frac{\partial}{\partial h_i}(-h_y + \log \overset{k}{\underset{j=1}{\sum}}exp(h_j)) \\
&= -1_{i=y} + \frac{\frac{\partial}{\partial h_i}(\underset{j=1}{\overset{k}{\sum}}exp(h_j))}{\underset{j=1}{\overset{k}{\sum}}exp(h_j)}  \\
&= -1_{i=y} + \frac{exp(h_i)}{\underset{j=1}{\overset{k}{\sum}}exp(h_j)}
\end{align}$$
hence
$$\nabla_hl_{ce}(h,y)=z-e_y\text{ }z=\text{normalize}(exp(h))$$
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006172044.9kg5xyojh4.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241006172447.67xg3l7z4r.webp)

