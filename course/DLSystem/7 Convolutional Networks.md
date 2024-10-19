---
date: 2024-10-19 14:57:38
date modified: 2024-10-19 16:18:34
title: Welcome to use Hexo Theme Keep
tags:
  - Hexo
  - Keep
categories:
  - Hexo
---
## Convolutional operators in deep networks

The reason that fully connected layer won't work very well besides just involving too many parameters that are going to overfit to the data, is that it doesn't capture the intuitive invariances we could expect to have in an image.

we want some sort of design for the architecture that preserves the kind of structure that we want that we have in the image.

Convolutions combine two ideas that are well-suited to processing images.

1. Require that activations between layers occur only in a "local" manner, and treat hidden layers themselves as spatial images

2. Share weights across all spatial locations.

Convolutions in deep networks are virtually always *multi-channel* convolutions:

map multi-channel (e.g. RGB) inputs to multi-channel hidden units

- $x\in\mathbb{R}^{h\times\ w\times c_{in}}$ denotes $c_{in}$ channel, size $h\times w$ image input
- $z\in\mathbb{R}^{h\times w\times c_{out}}$ denotes $c_{out}$ channel, size $h\times w$ image input
- $W\in\mathbb{R}^{c_{in}\times c_{out}\times k \times k}$ (order 4 tensor) denotes convolutional filter

we use a lot channels to represent kind of more complex and elements in hidden layer. So each channel is still this 2D spatial array, but we're going to use many of them to represent our hidden units.
$$
z[:,:,s] = \overset{c_{in}}{\underset{r=1}{\sum}}x[:,:,r]*W[r,s,:,:]
$$
 a more intuitive way to think about multi-channel convolutions: they are a generalization of traditional convolutions with scalar multiplications replaced by matrix-vector products.
 
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.7ljzk1g244.webp)

> 理解：每个输入的元素对应着一个$c_{in}$ 维度的向量，输出的元素对应着一个$c_{out}$维的向量，卷积核中的每个元素其实是一个矩阵，对应着 $\mathbb{R}^{c_{in}}\rightarrow\mathbb{R}^{c_{out}}$ 的映射。

## Elements of practical convolutions

### Padding

**Challenge**: "Naive" convolutions produce a smaller output than input image.

**Solution**: for (odd) kernel size $k$, pad input with $(k-1)/2$ zeros on all sides, results an output that is the same size as the input.

- Variants like circular padding, padding with mean values, etc.

### Strided Convolutions / Pooling

**Challenge**: Convolutions keep the same resolution of the input at each layer, don't naively allow for representations at different "resolutions".

**Solution #1**: incorporate max or average *pooling* layers to aggregate information.

**Solution #2**: slide convolutional filter over image in increments > 1(= stride)

**down sampling**

### Grouped Convolutions

**Challenge**: for large numbers of input/output channels, filters can still have a large number of weight, can lead to overfitting + slow computation

**solution**: Group together channels, so that groups of channels in output only depend on corresponding groups of channels in input (equivalently, enforce filter weight matrices to be block-diagonal)

**depth-wise convolution**

### Dilations

**Challenge**: Convolutions each have a relatively small receptive field size.

**Solution**: Dilate (spread out) convolution filter, so that it covers more of the image (see also: later architectures we will discuss, like self-attention layers); note that getting an image of the same size again requires adding more padding.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.m2fuxtki.webp)

## Differentiating convolutions

### What is needed to differentiate convolution?

we need to be able to multiply by its partial derivatives (adjoint operation)

if we define our operation

$$
z=\text{conv}(x,W)
$$

how do we multiply by the adjoints
$$
\bar{v}\frac{\partial \text{conv}(x,W)}{\partial W},\bar{v}\frac{\partial \text{conv}(x,W)}{\partial x}
$$
matrix-vector product operation 
$$
z=Wx\quad x\in\mathbb{R}^n,z\in\mathbb{R^m},W\in\mathbb{R}^{m\times n}
$$
Then $\frac{\partial z}{\partial x}=W$, so we need to compute the adjoint product
$$
\hat{v}^TW\iff W^T\bar{v}
$$
in other words, for a matrix vector multiply operation $Wx$, computing the backwards pass requires multiplying by the *transpose* $W^T$

what is the "transpose" of a convolution? 

> multiplying by the transpose of a convolution is equivalent to convolving with a flipped version of the filter.

$$
\bar{v}\frac{\partial\text{conv}(x,W)}{\partial x}=\text{conv}(\bar{v},\text{flip}(W))
$$
What about the other adjoint, $\bar{v}\frac{\partial \text{conv}(x,W)}{\partial W}$ ?

For this term, observe that we can also write the convolution as a matrix-vector product treating the *filter* as the vector

So adjoint requires multiplying by the transpose of the $x$-based matrix (actually a relatively practical approach, see future lecture on the "im2col" operation)

