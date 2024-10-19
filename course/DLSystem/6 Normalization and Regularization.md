---
date: 2024-10-18 16:24:36
date modified: 2024-10-19 14:23:37
title: Normalization and Regularization
tags:
  - course
categories:
  - DLSystem
---
## Normalization

Initialization matters a lot for training, and vary over the course of training to no longer be "consistent" across layers / networks.

### Layer normalization

normalize (mean zero and variance one) activations at each layer:

$$\begin{align}
&\hat{z}_{i+1}=\sigma_i(W^T_iz_i+b_i) \\
&z_{i+1} = \frac{\hat{z}_i-\mathbf{E}[\hat{z}_{i+1}]}{\sqrt{\mathbf{Var}[\hat{z}_{i+1}]}+\epsilon} 
\end{align}$$
$$\begin{align}
& \mathbf{E}[\hat{z}_{i+1}]=\frac{1}{n}\overset{n}{\underset{j=1}{\sum}}(\hat{z}_{i+1})_j\\
& \mathbf{Var}[\hat{z}_{i+1}]=\frac{1}{n}\overset{n}{\underset{j=1}{\sum}}((\hat{z}_{i+1})_j-\mathbf{E}[\hat{z}_{i+1}])^2
\end{align}$$

In practice, for standard FCN, harder to train resulting networks to low loss (relative norms of examples are a useful discriminative feature).

It's often harder to train networks to low loss when you add layer norm.

- layer norm is taking each individual example and forcing its norm of the activations, including the last layer(mean) to be equal to 0, (variances) equal to 1.

- maybe the relative norms of different examples are actually a pretty good feature to classify things.

### Batch normalization

let's consider the matrix form of our updates
$$
\hat{Z}_{i+1}=\sigma_i(Z_iW_i+b_i^T)
$$
then layer normalization is equivalent to normalizing the *rows* of this matrix.

What if, instead, we normalize it's columns? This is called batch normalization, as we are normalizing the activations over the *minibatch*.

### Minibatch dependence

Common solution is to compute a running average of mean/variance for all features at each layer $\hat{\mu}_{i+1}$, $\hat{\sigma}^2_{i+1}$, and at test time normalize by these quantities.
$$
(z_{i+1})_j=\frac{(\hat{z}_{i+1})-(\hat{mu}_{i+1})_j}{\sqrt{(\hat{\sigma}^2_{i+1})_j+\epsilon}}
$$

## Regularization

Regularization is the process of "limiting the complexity of the function class" in order to ensure that networks will generalize better to new data; typically occurs in two ways in deep learning.

- *Implicit regularization* refers to the manner in which our existing algorithms or architectures already limit functions considered.

- *Explicit regularization* refers to modifications made to the network and training procedure explicitly intended to regularize the network

###  $\mathcal{l}_2$ Regularization a.k.a. weight decay

$$
\underset{W_{1:L}}{\text{minimize}}\frac{1}{m}\overset{m}{\underset{i=1}{\sum}}l(h_{W_{1:L}}(x^{(i)}), y^{(i)}) + \frac{\lambda}{2}\overset{L}{\underset{i=1}{\sum}}||W_i||_f^2
$$

Results in the gradient descent updates:
$$
W_i:=W_i-\alpha\nabla_{W_i}\overset{m}{\underset{i=1}{\sum}}
l(h_{W_{1:L}}(x^{(i)}),y^{(i)})-\alpha\lambda W_i
=(1-\alpha\lambda)W_i-\alpha\nabla_{W_i}\overset{m}{\underset{i=1}{\sum}}
l(h_{W_{1:L}}(x^{(i)}),y^{(i)})
$$

I.e. at each iteration we shrink the weights by a factor $(1-\alpha\lambda)$ before taking the gradient step

### Dropout

Another common regularization strategy: randomly set some fraction of the activations at each layer to zero

$$\begin{align}
&\hat{z}_i=\sigma_i(W^T_iz_i+b_i)\\
&(z_{i+1})_j=
\begin{cases}
&(\frac{\hat{z}_{i+1}}{1-p}) \quad \text{with prob}\quad 1-p\\
&0\quad \text{with prob}\quad p
\end{cases}
\end{align}$$

Instructive to consider Dropout as bringing a similar stochastic approximation as SGD to the setting of individual activations.

$$\begin{align}
& \frac{1}{m}\overset{m}{\underset{i=1}{\sum}}l(h(x^{(i)}, y^{(i)}))\approx\frac{1}{|B|}\underset{i\in B}{\sum}l(h(x^{(i)},y^{(i)}))\\
&z_{i+1}=\sigma_i(\overset{m}{\underset{j=1}{\sum}}W_{j:}(z_i)_j)\approx z_{i+1}=\sigma_i(\underset{j\in P}{\frac{n}{|P|}\sum}W_{i:}(z_i)_j)
\end{align}$$

> I don't want to give the impression that deep learning is all about random hacks: there have been a lot of excellent scientific experimentation with all the above
> 
> But it is true that we don't have a complete picture of how all the different empirical tricks people use really work and interact 
> 
> The "good" news is that in many cases, it seems to be possible to get similarly good results with wildly different architectural and methodological choices.

