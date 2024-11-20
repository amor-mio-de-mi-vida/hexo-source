---
date: 2024-11-18 18:37:27
date modified: 2024-11-18 22:58:52
title: Welcome to use Hexo Theme Keep
tags:
  - PCA
  - ML
categories:
  - Machine Learning
date created: 2024-09-25 13:26:08
---
## Goal of PCA

- Compute the most meaningful basis to re-express a noisy data set

- Hope that this new basis will filter out the noise and reveal hidden structure.

## Dimensionality Reduction

Suppose that we have $m$ samples with features of $n$ dimension: 
$$\left(\begin{matrix}
\mathbf{x_1}\\
\mathbf{x_2}\\
\vdots\\
\mathbf{x_m}
\end{matrix}\right)=\left(\begin{matrix}
x_{11}&x_{12}&\cdots&x_{1n}\\
x_{21}&x_{22}&\cdots&x_{2n}\\
\vdots&\vdots&\cdots&\vdots\\
x_{m1}&x_{m2}&\cdots&x_{mn}\\
\end{matrix}\right)\in\mathbb{R}^{m\times n}$$
We want to find a line, which we project our samples to, we get the maximum variance of the all projections. Or we want the minimum reconstruction error. Suppose the direction of the line is $\mathbf{v}$, we have $||\mathbf{v}|| = 1$.

We re-center all our samples and we get:
$$\mathbf{A}=\begin{align}
\left(\begin{matrix}
\mathbf{x_1-\bar{x}_1}\\
\mathbf{x_2-\bar{x}_2}\\
\vdots\\
\mathbf{x_m-\bar{x}_m}
\end{matrix}\right)\in\mathbb{R}^{m\times n}
\end{align}$$

### Minimum reconstruction cost

The reconstruction cost is 
$$\text{cost}=\overset{m}{\underset{i=1}{\sum}}(||x_i-\bar{x}_i||^2-||(x_i-\bar{x}_i)v||^2)$$
since that $\overset{m}{\underset{i=1}{\sum}}||x_i-\bar{x}_i||^2$ is fixed we just need to maximize
$$\overset{m}{\underset{i=1}{\sum}}||\mathbf{(x_i-\bar{x}_i)}\cdot\mathbf{v}||^2$$
which is the variance of projections. So the minimum reconstruction cost can convert to Maximum variance.

### Maximum variance

After re-centering all our samples (subtract the mean of each row), the projection of $\mathbf{x_i}$ is $\mathbf{x_i}\cdot\mathbf{v}$, so the objective function is :
$$\begin{align}
\text{var} &= \underset{\mathbf{v}}{argmax}(\overset{m}{\underset{i=1}{\sum}}||\mathbf{(x_i-\bar{x}_i)}\cdot\mathbf{v}||^2) \\ 
&=\underset{\mathbf{v}}{argmax}(\overset{m}{\underset{i=1}{\sum}}\mathbf{v}^T(\mathbf{x_i-\bar{x}_i})^T\mathbf{(x_i-\bar{x}_i)}\mathbf{v})
\end{align}$$

Then we have
$$\begin{align}
\text{var}&=\underset{\mathbf{v}}{argmax}(\mathbf{v}^T\overset{m}{\underset{i=1}{\sum}}(\mathbf{x-\bar{x}_i})^T(\mathbf{x-\bar{x}_i})\mathbf{v})\\
&=\underset{\mathbf{v}}{argmax}(\mathbf{v}^T\mathbf{A}^T\mathbf{A}\mathbf{v})
\end{align}$$
we find that $\mathbf{A}^T\mathbf{A}$ is the covariance matrix.
$$\mathbf{A}^T\mathbf{A}=\left(\begin{matrix}
\mathbf{(x_1-\bar{x}_1)^2}&\mathbf{(x_1-\bar{x}_1)(x_2-\bar{x}_2)}&\cdots&\mathbf{(x_1-\bar{x}_1)(x_m-\bar{x}_m)}\\
\mathbf{(x_2-\bar{x}_2)(x_1-\bar{x}_1)}&\mathbf{(x_2-\bar{x}_2)^2}&\cdots&
\mathbf{(x_2-\bar{x}_2)(x_2-\bar{x}_m)}\\
\vdots&\vdots&\cdots&\vdots\\
\mathbf{(x_m-\bar{x}_m)(x_1-\bar{x}_1)}&\mathbf{(x_m-\bar{x}_m)(x_2-\bar{x}_2)}&\cdots&\mathbf{(x_m-\bar{x}_m)^2}
\end{matrix}\right)$$

due to Singular Value Decomposition theorem, we have 
$$A=U\Sigma V^T$$
$U\in\mathbb{R}^{m\times m}$ is a unitary matrix, $\Sigma\in\mathbb{R}^{m\times n}$ is a rectangular diagonal matrix with non-negative real numbers on the diagonal, $V\in\mathbb{R}^{n\times n}$ is a unitary matrix.
$$
V^T(A^TA)V=\text{diag}(\lambda_1,\lambda_2,...,\lambda_n)
$$
$$
\Rightarrow(A^TA)V=\text{diag}(\lambda_1,\lambda_2,...,\lambda_n)V
$$

so we have 
$$\begin{align}
\text{var}&=\underset{\mathbf{v}}{argmax}(v^TA^TAv)\\
&=\underset{\mathbf{v}}{argmax}(v^TV\Sigma^TU^TU\Sigma V^Tv)\\
&=\underset{\mathbf{v}}{argmax}(||\Sigma V^Tv||^2)\end{align}$$

if $v$ is one of the eigenvectors of $A^TA$, we get the max variance, and then, we have

$$A^TAv=\lambda v$$
so 
$$v^TA^TAv=v^T\lambda v=\lambda$$
The max variance equals to the max eigenvalue of $A^TA$. The direction vector of the projected line is $v$.

> the variance of $\lambda$ represent the information preserved after the projection.

## PCA algorithm

**Step1: Start from $m\times n$ data matrix $A$.**

- $m$ data points (samples over time)

- $n$ measurement types

**Step2: Re-center: subtract mean from each row of $A$.**

**Step3: Compute covariance matrix: $\Sigma=A^TA$**

**Step4: Compute eigenvectors and eigenvalues of $\Sigma$**

**Step5: Principal components: $k$ eigenvectors with highest eigenvalues.**

### Efficient Computation of Eigenvectors

If $A\in\mathbb{R}^{m\times n}$ and $m<< n$, then $\Sigma=A^TA\in\mathbb{R}^{n\times n}$, (m is the number of images, n is the number of features).

use $AA^T$ instead, eigenvector of $AA^T$ is easily converted to that of $A^TA$. If $y$ is the eigenvector of $AA^T$, then
$$(AA^T)y=\lambda y$$
so
$$\begin{align}
&A^T(AA^T)y=\lambda (A^Ty)\\
&(A^TA)(A^TA)y=\lambda(A^TA)
\end{align}$$
so $A^Ty$ is the eigenvector of $A^TA$

## Recognition with Eigenfaces

Step1: Process the image database (set of images with labels)

- Run PCA —— compute eigenfaces

- Calculate the K coefficients for each image

Step2: Given a new image (to be recognized)$x$, calculate K coefficients

Step3: Detect if $x$ is a face
$$
x\rightarrow(a_1,a_2,...,a_k)
$$
Step4: If x is a face, who is it?
$$
||\mathbf{x}-(\mathbf{\bar{x}}+a_1\mathbf{v_1}+a_2\mathbf{v_2}+...+a_k\mathbf{v_k})||<\text{threshold}
$$
Find closest labeled face in database (nearest-neighbor in K-dimensional space).

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/其他/3931a9abc5dfa8388a27b29ffab7049.ic5bkyvv1.webp)

## Singular Value Decomposition SVD

The eigenvectors of a matrix $\mathbf{A}$ form a basis for working with $\mathbf{A}$. However, for rectangular matrix $\mathbf{A}(m\times n)$, $\text{dim}(\mathbf{A}x)\ne\text{dim}(x)$ and the concept of eigenvectors does not exist.

Yet $\mathbf{A}^T\mathbf{A}(n\times n)$\footnote{here each row of $\mathbf{A}$ is a measurement in time and each column is a measurement type} is a symmetric, real matrix($\mathbf{A}$ is real) and therefore, there is an orthonormal basis of eigenvectors $\{u_k\}$ for $\mathbf{A}^T\mathbf{A}$.

Consider the vectors $\{v_k\}$

$$v_k=\frac{\mathbf{A}u_k}{\sqrt{\lambda_k}}$$

They are also orthonormal, since:

$$u_j^T\mathbf{A}^T\mathbf{A}u_k=\lambda_k\delta(k-j)$$

Since $\mathbf{A}^T\mathbf{A}$ is positive semidefinite, its eigenvalues are non-negative $\{\lambda_k\ge 0\}$

Define the singular values of $\mathbf{A}$ as

$$\sigma_k=\sqrt{\lambda_k}$$

and order them in a non-increasing order:

$$\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_n\ge 0$$

Motivation: One can see, that if $\mathbf{A}$ itself is square and symmetric, then $\{u_k,\sigma_k\}$ are the set of its own eigenvectors and eigenvalues.

For a general matrix $\mathbf{A}$, assume $\{\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_R > 0=\sigma_{r+1}=\sigma_{r+2}=\cdots =\sigma_n\}$

$$\mathbf{A}u_k=0\cdot v_k , \quad k=r+1,...,n, u_k\in \mathbb{R}^{n\times 1}, v_k\in\mathbb{R}^{m\times 1}$$

Now we can write:

$$\begin{align}

&\left[\begin{matrix}

|&&|&|&&|\\

\mathbf{A}u_1&\cdots&\mathbf{A}u_r&\mathbf{A}u_{r+1}&\cdots&\mathbf{A}u_n\\

|&&|&|&&|\\

\end{matrix}\right]=\mathbf{A}

\left[\begin{matrix}

|&&|&|&&|\\

u_1&\cdots&u_r&u_{r+1}&\cdots&u_n\\

|&&|&|&&|\\

\end{matrix}\right]\\

&=\mathbf{AU}=\left[\begin{matrix}

|&&|&|&&|\\

\sigma_1v_1&\cdots&\sigma_rv_r&0\cdot v_{r+1}&\cdots&0\cdot v_n\\

|&&|&|&&|\\

\end{matrix}\right]\\

&=\left[\begin{matrix}

|&&|&|&&|\\

v_1&\cdots&v_r&v_{r+1}&\cdots&v_n\\

|&&|&|&&|\\

\end{matrix}\right]

\left[\begin{matrix}

\sigma_1&\cdots&0&0&\cdots&0\\

\vdots&\cdots&\vdots&\vdots&\cdots&\vdots\\

0&\cdots&\sigma_r&0&\cdots&0\\

0&\cdots&0&0&\cdots&0\\

\vdots&\cdots&\vdots&\vdots&\cdots&\vdots\\

0&\cdots&0&0&\cdots&0\\

\end{matrix}\right]=\mathbf{V\Sigma}

\end{align}$$

$$AUU^T=V\Sigma U^T\Rightarrow A=V\Sigma U^T$$

$$A\in\mathbb{R}^{m\times n},V\in\mathbb{R}^{m\times m},\Sigma\in\mathbb{R}^{m\times n},U\in\mathbb{R}^{n\times n}$$

while $V$ and $U$ are both orthogonal matrix.

> 正交矩阵的性质：
> 
> **定义** 设n阶矩阵$A$满足$AA^T=A^TA=I$, 则称A为正交矩阵
> 
> **定理1** 设$A,B$是同阶正交矩阵，则：
> - $\det(A)=\pm1$
> - $A^T$, $A^{-1}$, $A^{*}$ 均为正交矩阵
> - $AB$为正交矩阵
> 
> **定理2** 实方阵$A$为正交矩阵$\iff A$的列/行向量组为标准正交向量组
> 
>**定理3** （正交变换的保范性）设$A$为正交矩阵, 则$\forall x_1,x_2\in\mathbb{R}^n$，有
>- $<Ax_1, Ax_2>=<x_1,x_2>$;
>- $||Ax_1||=||x_1||$
>
>**定理4** 设$A$为正交矩阵，则$A$的特征值只能为$\pm1$
>
>**定理5** 设$A$为n阶实对称矩阵，则**一定**存在n阶正交矩阵P使得
>$$P^{-1}AP=\text{diag}(\lambda_1,\lambda_2,...,\lambda_n)$$

