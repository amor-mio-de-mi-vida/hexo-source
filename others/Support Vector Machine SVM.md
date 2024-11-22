---
date: 2024-11-22 08:59:05
date modified: 2024-11-22 10:16:55
title: Support Vector Machine SVM
tags:
  - SVM
  - ML
categories:
  - Machine Learning
date created: 2024-09-25 13:26:08
---
Reference:

https://blog.csdn.net/weixin_48228548/article/details/124133393

https://zhuanlan.zhihu.com/p/38182879

https://zhuanlan.zhihu.com/p/38182879

<!--more-->

## Hard Margin SVM
$$\begin{align}
\begin{cases}
&\max\text{ margin}(\omega_1,b)\\
&\text{s.t. }\begin{cases}
\omega^Tx_i+b>0,\quad y_i>0\\
\omega^Tx_i+b<0,\quad y_i<0\\
\end{cases}\\
&\rightarrow y_i(\omega^Tx_i+b)>0\\
&\text{margin}(\omega,b)=\underset{\omega,b}{\min}\frac{|\omega^Tx_i+b|}{||\omega||}
\end{cases}
\end{align}$$
so we have 
$$\begin{align}&\max \underset{\omega,b}{\min}\frac{1}{||\omega||}(\omega^Tx_i+b)y_i\quad\text{for }x_i,i=1,2,...n\\
&\text{s.t. }y_i(\omega^Tx_i+b)>0
\end{align}$$
there exists a $\gamma>0$ s.t. all distance > $\gamma$.
$$\begin{align}
&=\underset{\omega,b}{\max}\frac{1}{||\omega||}\underset{x_i}{\min}y_i(\omega^Tx+b)\\ 
&\exists\gamma>0, s.t. \min y_i(\omega^Tx_i+b)=r
\end{align}$$
which implies
$$\begin{align}
&\underset{\omega,b}{\max}\frac{1}{||\omega||}\\
&\text{s.t. }\min y_i(\omega^Tx+b)=1
\end{align}$$
so we have the final objective function
$$\begin{align}
&\underset{\omega,b}{\min}\frac{1}{2}||\omega||^2\\
&\text{s.t. }y_i(\omega^Tx_i+b)\ge 1,i=1,2,...,m
\end{align}$$
Due to Lagrange function, the objective function can convert to
$$\begin{cases}
&\underset{\omega,b}{\min}\underset{\lambda}{\max}L(\omega,b,\lambda)\\ 
&\text{s.t. }\lambda_i\ge0,\underset{i=0}{\overset{N}{\sum}}\lambda_iy_i=0
\end{cases}$$
Lagrange function is convex, it is not easy to get its maximum, so that we convert to its dual problem. ($\min\max f(x)=\max\min f(x)$)
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/其他/image.9gwlovpb01.webp)
so, we have:
$$\begin{cases}
\underset{\lambda}{\max}\underset{\omega,b}{\min}L(\omega,b,\lambda)\\
\text{s.t. }\lambda_i\ge 0,\overset{N}{\underset{i=0}{\sum}}\lambda_iy_i=0
\end{cases}$$
so that we can get the minimum of Lagrange Function
$$\begin{align}
&\min L(\omega,b,\lambda)=\frac{1}{2}\omega^T\omega+\overset{N}{\underset{i=1}{\sum}}\lambda_i(1-y_i(\omega^Tx_i+b))\\
&y_i\in\{1,-1\},\quad y_i(\omega^Tx_i+b)\ge1
\end{align}$$
$$\text{let }\frac{\partial L}{\partial b}=0\Rightarrow-\overset{N}{\underset{i=1}{\sum}}\lambda_iy_i=0$$
$$\begin{align}
L(\omega,b,\lambda)&=\frac{1}{2}\omega^T\omega+\overset{N}{\underset{i=1}{\sum}}\lambda_i-\overset{N}{\underset{i=1}{\sum}}\lambda_iy_i\omega^Tx_i-\overset{N}{\underset{i=1}{\sum}}\lambda_iy_ib\\
&=\frac{1}{2}\omega^T\omega+\overset{N}{\underset{i=1}{\sum}}\lambda_i-\overset{N}{\underset{i=1}{\sum}}\lambda_iy_i\omega^Tx_i
\end{align}$$
$$\text{let }\frac{\partial L}{\partial \omega}=0\Rightarrow\omega-\underset{i=1}{\overset{i=1}{\sum}}\lambda_iy_ix_i=0\Rightarrow\omega=\overset{N}{\underset{i=1}{\sum}}\lambda_iy_ix_i$$
$$\begin{align}
L(\omega,b,\lambda)&=\frac{1}{2}(\overset{N}{\underset{i=1}{\sum}}\lambda_iy_ix_i)^T\overset{N}{\underset{j=1}{\sum}}\lambda_jy_jx_y-\overset{N}{\underset{i=1}{\sum}}\lambda_iy_i(\overset{N}{\underset{j=1}{\sum}}\lambda_jy_jx_j)^Tx_i+\underset{i=1}{\overset{N}{\sum}}\lambda_i\\
&=\frac{1}{2}\overset{N}{\underset{i=1}{\sum}}\underset{j=1}{\overset{N}{\sum}}\lambda_iy_ix_i^T\lambda_jy_jx_j-\underset{i=1}{\overset{N}{\sum}}\underset{j=1}{\overset{N}{\sum}}\lambda_iy_i\lambda_jy_jx_j^Tx_i+\underset{i=1}{\overset{N}{\sum}}\lambda_i\\
&=\underset{=1}{\overset{N}{\sum}}\lambda_i-\frac{1}{2}\underset{i=1}{\overset{N}{\sum}}\underset{j=1}{\overset{N}{\sum}}\lambda_iy_i\lambda_jy_jx_j^Tx_i
\end{align}$$
thus, we have the final object function
$$\begin{align}
&\underset{\alpha}{\max}\overset{m}{\underset{i=1}{\sum}}\alpha_i-\underset{i=1}{\overset{m}{\sum}}\underset{j=1}{\overset{m}{\sum}}\alpha_iy_i\alpha_jy_jx_j^Tx_i\\
&s.t.\quad \overset{m}{\underset{i=1}{\sum}}\alpha_iy_i=0,\alpha_i\ge0,\quad i=1,2,...,m
\end{align}$$
Solution to the dual form, we introduce KKT theorem:
$$\begin{cases}
\alpha_i\ge0;\\
y_if(x_i)-1\ge0;\\
\alpha_i(y_if(x_i)-1)=0
\end{cases}$$

$\omega^*$ 的求解： 直接带入$\partial L(\lambda,\omega,b)/\partial\lambda=0$, 条件即可。

$b^*$的求解：由于仅有支持向量上的点起作用，所以代回支持向量上的样本点，对$b^*$进行求解。
$$\text{let }\frac{\partial L}{\partial\lambda}=0\Rightarrow\omega^*=\overset{N}{\underset{i=0}{\sum}}\lambda_iy_ix_i$$

$$\begin{align}
\exists(x_k,y_k)\quad s.t.&\\
&\quad1-y_k(\omega^Tx_k+b)=0\\
&\Rightarrow y_k(\omega^Tx_k+b)=1\\
&\Rightarrow y_k^2(\omega^Tx_k+b)=y_k\\
&\Rightarrow (\omega^Tx_k+b)=y_k
\end{align}$$
so 
$$\begin{align}b^*&=y_k-\omega^Tx_k\\
&=y_k-\overset{N}{\underset{i=0}{\sum}}\lambda_iy_ix_i^Tx_k
\end{align}$$
finally we have
$$\begin{cases}
\omega^*=\overset{N}{\underset{i=0}{\sum}}\lambda_iy_ix_i\\
b^*=y_k-\overset{N}{\underset{i=0}{\sum}}\lambda_iy_ix_i^Tx_k
\end{cases}$$

## Soft Margin SVM

$$\begin{align}
&\underset{\omega,b,\xi_i}{\min}\frac{1}{2}||\omega||^2+C\overset{m}{\underset{i=1}{\sum}}\xi_i\\
&\xi_i\ge0,\quad i=1,2,...,m
\end{align}$$


## KKT Theorem


## Lagrange Function






