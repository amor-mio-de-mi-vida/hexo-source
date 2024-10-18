---
date: 2024-10-17 15:10:25
date modified: 2024-10-18 10:25:12
title: Optimization
tags:
  - course
categories:
  - DLSystem
date created: 2024-09-25 13:26:08
---
## Fully connected networks

A L-layer, fully connected network, MLP, now with an explicit bias term, is defined by the iteration
$$\begin{align}
&z_{i+1} = \sigma_i(W^T_iz_i+b_i), i=1,\cdots,L\\
&h_{\theta}(x)\equiv z_{L+1}\\
&z_1 \equiv x
\end{align}$$
Where parameters $\theta=\{W_{1:L},b_{1:L}\}$, and where $\sigma_i(x)$
 is the nonlinear activation, usually with $\sigma_L(x)=x$
$$Z_i\in\mathbb{R}^{n_i},W_i\in\mathbb{R}^{n_i\times n_{i+1}}, b_i\in\mathbb{R}^{n_{i+1}}$$

### Broadcast


### Key questions for fully connected networks

- How do we choose the width and depth of the network?

- How do we actually optimize the objective ("SGD" is the easy answer, but not the algorithm most commonly used in practice)

- How do we initialize the weights of the network?

- How do we ensure the network can continue to be trained easily over multiple optimization iterations?

## Optimization

### Newton's Method

One way to integrate more "global" structure into optimization methods is Newton's method, which scales gradient according to inverse of the Hessian (matrix of second derivatives)
$$
\theta_{t+1}=\theta_t-\alpha(\nabla_{\theta}^2f(\theta_t))^{-1}\nabla_{\theta}f(\theta_t)
$$
where $\nabla^2_{\theta}f(\theta_t)$ is the *Hessian*, $n\times n$matrix of all second derivatives.

Full step given by $\alpha=1$,otherwise called a damped Newton method.
$$\begin{align}
&f:\mathbb{R}^{n}\rightarrow\mathbb{R}\\
&\nabla_{\theta}^2f(\theta)=
\left [
\begin{matrix}
&\frac{\partial^2f(\theta)}{\partial\theta_1^2}&\cdots&\frac{\partial^2f(\theta)}{\partial \theta_1\partial\theta_n} \\
&\cdots&\cdots&\cdots\\
&\frac{\partial^2f(\theta)}{\partial\theta_n\partial\theta_1}&\cdots&\frac{\partial^2f(\theta)}{\partial\theta_n^2}
\end{matrix}
\right ]
\end{align}
$$

### Momentum

$$\begin{align}
&u_{t+1}=\beta u_t+(1-\beta)\nabla_\theta f(\theta_t)\\
&\theta_{t+1} = \theta_t-\alpha u_{t+1}
\end{align}$$
to "unbias" the update to have equal expected magnitude across all iterations.
$$
\theta_{t+1}=\theta_t-\frac{\alpha u_{t+1}}{1-
\beta^t}$$
### Nesterov Momentum

$$\begin{align}
&u_{t+1}=\beta u_t+(1-\beta)\nabla_{\theta}f(\theta_t-\alpha u_t)\\
&\theta_{t+1}=\theta_t-\alpha u_{t-1}
\end{align}$$
### Adam

Most widely used adaptive gradient method for deep learning is Adam algorithm, which combines momentum and adaptive scale estimation.

$$\begin{align}
&u_{t+1} = \beta_1u_t+ (1-\beta_1)\nabla_\theta f(\theta_t)\\
&v_{t+1} = \beta_2v_t+(1-\beta_2)(\nabla_\theta f(\theta_t)\odot\nabla_\theta f(\theta_t))\\
&\theta_{t+1}=\theta_t-\frac{\alpha u_{t+1}}{\sqrt{v_{t+1}}+\epsilon}
\end{align}$$

we're taking each component of our gradient, we're scaling it by its magnitude that puts all the gradient terms onto a similar scale.

### Stochastic Gradient Descent

repeating for batches $B\subset\{1,\cdots,m\}$
$$
\theta_{t+1}=\theta_t-\frac{\alpha}{|B|}\underset{i\in B}{\sum}\nabla_\theta l(h_\theta(x^{(i)}, y^{(i)}))
$$

You need to constantly experiment to gain an understanding / intuition of how these methods actually affect deep networks of different types.

## Initialization

### Initialization of weights

Recall that we optimize parameters iteratively by stochastic gradient descent

$$
W_i :=W_i-\alpha\nabla_{W_i}l(h_\theta(X),y)
$$

But how do we choose the *initial* values of $W_i$, $b_i$ ?

Recall the manual backpropagation forward/backward passes (without bias):

$$\begin{align}
&Z_{i+1}=\sigma_i(Z_iW_i)=0 \\
&G_i= (G_{i+1}\odot\sigma^{'}(Z_iW_i))W_i^T=0
\end{align}$$

- if $W_i=0$, then $G_j=0\text{for}j\le i,\Rightarrow \nabla_{W_i}l(h_\theta(X),y)=0$

- I.e., $W_i=0$ is a (very bad) local optimum of the objective

### Weights don't move "that much"

weights often stay much closer to their initialization than to the "final" point after optimization from different.

End result: initialization matters ...

### What causes these effects?

Consider independent random variables $x\sim \mathcal{N}(0,1), w\sim\mathcal{N}(0,\frac{1}{n})$; then
$$
\mathbf{E}[x_iw_i]=\mathbf{E}[x_i]\mathbf{E}[w_i]=0, \mathbf{Var}[x_iw_i]=\mathbf{Var}[x_i]\mathbf{Var}[w_i]=\frac{1}{n}
$$
so $\mathbf{E}[w^Tx]=0,\mathbf{Var}[w^Tx]=1(w^Tx\rightarrow\mathcal{N}(0,1)\text{ by central limit theorem)}$

Thus, informally speaking if we used a linear activation and $z_i \sim \mathcal{N}(0,I),W_i\sim\mathcal{N}(0,\frac{1}{n}I)$,then $z_{i+1}=w_i^Tz_i\sim \mathcal{N}(0,I)$

If we use a ReLU nonlinearity, then "half" the components of $z_i$ will be set to zero, so we need twice the variance on $W_i$ to achieve the same final variance, hence $W_i\sim\mathcal{N}(0,\frac{2}{n}I)$ (Kaiming normal initialization)




