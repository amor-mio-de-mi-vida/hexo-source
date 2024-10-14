---
title: Manual Neural Networks
tags:
  - course
categories:
  - DLSystem
date: 2024-10-07 20:11:45
date modified: 2024-10-14 21:22:13
---
## From linear to nonlinear hypothesis classes

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241007202008.7w6t0ry9b2.webp)
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241007201939.6wqpnlvi59.webp)
How can we create the feature function $\phi$
- Through manual engineering of features relevant to the problem (the "old" way of doing machine learning)
- In a way that itself is learned from data (the "new" way of doing ML)
$$\phi (x)=\sigma (W^Tx)$$
where $W\in R^{n\times d}$ and $\sigma: \mathbb{R}^d\rightarrow \mathbb{R}^d$ is essentially any nonlinear function.
> when we essentially apply almost any non-linear function as our choice of sigma, we basically allow ourselves to get much richer representations than we can with this linear classifier.


$$h_{\theta}(x)=\theta^T\sigma(W^Tx)$$
Neural networks effectively, in this interpretation, can be viewed as a way of extracting features of our data in a manner where we train simultaneously both the final linear classifier as well as all the parameters of the feature vector itself.

## Neural networks
A neural network refers to a particular type of hypothesis class consisting of multiple, parameterized differentiable functions composed together in any manner to from the output.

### two layer neural network
$$\begin{align}
&h_{\theta}(x) = W_2^T\sigma (W_1^Tx) \\
&\theta ={W_1\in \mathbb{R}^{n\times d}, W_2\in \mathbb{R}^{d\times k}}
\end{align}$$
where $\sigma :\mathbb{R}\rightarrow \mathbb{R}$ is a nonlinear function applied elementwise to the vector
Written in batch matrix form
$$h_{\theta}(X)=\sigma(XW_1)W_2$$

### Universal function approximation
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241007205212.9gwk08vgrk.webp)
Assume one-hidden-layer ReLU network:
$$\hat{f}(x)=\overset{d}{\underset{i=1}{\sum}}\pm max\{0, w_ix + b_i\}$$
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/Pasted-image-20241007211543.9nzrvohm75.webp)
Waiting for replenishment....

### Multi-layer perceptron

In batch form

$$ 
\begin{align}
&Z_{i+1}=\sigma_i(Z_iW_i),i=1,...,L \\
&Z_1=X,\\
&h_{\theta}(X)=Z_{L+1}\\
&[Z_i \in \mathbb{R}^{m\times n_i},W_i\in \mathbb{R}^{n_i\times n_{i+1}}]
\end {align} 
$$


with nonlinearities $\sigma_i:\mathbb{R}\rightarrow \mathbb{R}$ applied elementwise, and parameters
$$\theta={W_1,...,W_L}$$

Can also optionally add bias term

## Backpropagation 

we want to solve the optimization problem 
$$\underset{\theta}{minimize}\frac{1}{m}\underset{i=1}{\overset{m}{\sum}}l_{ce}(h_{\theta}x^{(i)},y^{(i)})$$
using SGD, just with  $h_{\theta}(x)$  now being a neural network.
Requires computing the gradients $\nabla l_{ce}(h_{\theta}(x^{(i)}), y^{(i)})$ for each element of $\theta$
As for two-layer network, written in batch matrix form.
$$\nabla_{W_1,W_2}l_{ce}(\sigma(XW_1)W_2, y)$$
The gradient $W_2$ looks identical to the softmax regression case:
$$\begin{align}
\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial W_2}&=\frac{\partial l_{ce}(\partial(XW_1)W_2,y)}{\partial \sigma(XW_1)W_2}\otimes\frac{\partial \sigma(XW_1)W_2}{W_2} \\
&= (S-I_y)\otimes \sigma (XW_1)
\end{align}$$

$$S=\text{normalize}(exp(h_{\theta}(x)))=\text{normalize}(exp(\sigma (XW_1)W_2)) $$
so the gradient is 
$$
\nabla_{W_2}l_{ce}(\sigma(XW_1)W_2,y)=\sigma(XW_1)^T(S-I_y)
$$

as for $W_1$
$$\begin{align}
\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial W_1} &= \frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial \sigma(XW_1)W_2}\otimes\frac{\partial \sigma(XW_1)W_2}{\partial \sigma(XW_1)}\otimes\frac{\partial \sigma(XW_1)}{\partial XW_1}\otimes\frac{\partial XW_1}{\partial W_1} \\
&=(S-I_y)\otimes W_2 \otimes\sigma^{'}(XW_1)\otimes X
\end{align}$$
so the gradient is 
$$
\nabla_{W_1}l_{ce}(\sigma(XW_1)W_2,y)=X^T(\sigma^{'}(XW_1)\odot(S-I_y)W_2^T)
$$
as for fully-connected network
$$Z_{i+1}=\sigma_i(Z_iW_i),i=1,...,L$$
we can find that 
$$\begin{align}
G_i&=G_{i+1}\otimes\frac{\partial Z_{i+1}}{\partial Z_i}\\
&=G_{i+1}\otimes \frac{\partial\sigma(Z_iW_i)}{\partial Z_iW_i}\otimes \frac{\partial Z_iW_i}{\partial Z_i}\\
&=G_{i+1}\otimes\sigma^{'}(Z_iW_i)\otimes W_i
\end{align}$$
$$Z_i\in\mathbb{R}^{m\times n_i}$$
$$\begin{align}
G_i&=\frac{\partial l(Z_{L+1}, y)}{\partial Z_i}\\
&=G_{i+1}\otimes\sigma^{'}(Z_iW_i)\otimes W_i\\
&=(G_{i+1}\odot\sigma(Z_iW_i))
\end{align}$$
Similar formula for actual parameter gradients $\nabla_{W_i}l(Z_{L+1},y)\in \mathbb{R}^{n_i\times n_{i+1}}$
$$\begin{align}
\frac{\partial l(Z_{L+1},y)}{\partial W_i}&=G_{i+1}\otimes\frac{\partial Z_{i+1}}{\partial W_1}\\
&=G_{i+1}\otimes\frac{\partial\sigma(Z_iW_i)}{\partial Z_iW_i}\otimes \frac{\partial Z_iW_i}{\partial W_i} \\
&=G_{i+1}\otimes\sigma^{'}(Z_iW_i)\otimes Z_i
\end{align}$$
$$
\Rightarrow \nabla_{W_i}l(Z_{L+1},y)=Z_i^T(G_{i+1}\odot\sigma^{'}(Z_iW_i))
$$
### Backpropagation: Forward and backward passes
we can efficiently compute all the gradients we need for a neural network by following the prodcedure. 
1. Initialize: $Z_1$ = $X$
   Iterate: $Z_{i+1}=\sigma_i(Z_iW_i),i=1,...,L$
2. Initialize: $G_{L+1}=\nabla_{Z_{L+1}}l(Z_{L+1},y)=S-I_y$
   Iterate: $G_i=(G_{i+1}\odot\sigma^{'}_i(Z_iW_i))W_i^T,i=L,...,1$
And we can compute all the needed gradients along the way
$$
\nabla_{W_i}l(Z_{k+1},y)=Z_i^T(G_{i+1}\odot\sigma_i^{'}(Z_iW_i))
$$
"vector Jacobian product"s
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.3d4s3f3igf.webp)
