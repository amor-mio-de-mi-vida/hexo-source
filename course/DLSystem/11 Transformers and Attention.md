---
date: 2024-10-29 16:29:31
date modified: 2024-11-01 18:46:11
title: Welcome to use Hexo Theme Keep
tags:
  - course
categories:
  - DLSystem
---
## The two approaches to time series modeling

time series prediction task is the task of predicting 
$$
y_{1:T}=f_\theta(x_{1:T})
$$
where $y_t$ depend only on $x_{1:t}$

There are multiple methods for doing so, which may or may not involve the latent state representation of RNNs.

### The RNN "latent state" approach

We have already seen the RNN approach to time series: maintain "latent state" $h_t$ that summaries *all* information up until that point

**Pros**: Potentially "infinite" history, compact representation

**Cons**: Long "compute path" between history and current time $\Rightarrow$ vanishing / exploding gradients, hard to learn.

### The "direct prediction" approach

In contrast can also *directly* predict output $y_t$

$$
y_t = f_\theta(x_{1:t})
$$

(just need a function that can make predictions of differently-sized inputs)

**Pros**: Often can map from past to current state with shorter compute path

**Cons**: No compact state representation, finite history in practice.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.7i0e2k79ry.webp)

### CNNs for direct prediction

One of the most straightforward ways to specify the function $f_\theta:$ fully convolutional networks, a.k.a temporal convolutional networks (TCNs).

The main constraint is that the convolutions be causal: $z_t^{(i+1)}$ can only depend on $z_{t-k:t}^{(i)}$.

Many successful applications: e.g. WaveNet for speech generation (van den Oord et al., 2016)

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.2h8bbjtnrq.webp)

### Challenges with CNNs for dense prediction

Despite their simplicity, CNNs have a notable disadvantage for time series prediction: the *receptive field* of each convolution is usually relatively small $\Rightarrow$ need deep networks to actually incorporate past information.

Potential solutions:

- Increase kernel size: also increases the parameters of the network

- Pooling layers: not as well suited to dense prediction, where we want to predict all of $y_{1:T}$

- Dilated convolutions: "Skips over" some past state / inputs.

## Self-attention and Transformers

"Attention" in deep networks generally refers to any mechanism where individual states are *weighted* and then *combined*.

Instead of using just the last hidden layer, we take all our hidden states and we are going to combine them according to some weights.

$$z_t=\theta^Th_t^{(k)}$$
$$w=\text{softmax}(z)$$
$$\bar{h}=\underset{t=1}{\overset{T}{\sum}}w_th_t^{(k)}$$
Used originally in RNNs when one wanted to combine latent states over all times in a more general manner than "just" looking at the last state.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.1e8m0orvi8.webp)

### The self-attention operation

Self-attention refers to a particular form of attention mechanism 

Given three inputs $K$, $Q$, $V\in\mathbb{R}^{T\times d}$ ("keys", "queries", "values", in one of the least-meaningful semantic designations we have in deep learning)

$$K=
\left[\begin{matrix}
——k_1^T——\\
——k_2^T——\\
\vdots\\
——k_T^T——
\end{matrix}\right],
Q=
\left[\begin{matrix}
——q_1^T——\\
——q_2^T——\\
\vdots\\
——q_T^T——
\end{matrix}\right],
V=
\left[\begin{matrix}
——v_1^T——\\
——v_2^T——\\
\vdots\\
——v_T^T——
\end{matrix}\right],
$$

$$k=XW_k$$

we define the self attention operations as 

$$\text{SelfAttention}(K,Q,V)=\text{softmax}(\frac{KQ^T}{\sqrt{d}})V$$
$$KQ^T=(k_i^Tq_j)_{ij}$$
this entry corresponds in some sense to computing the similarity between the $k_i$ term and the $q_j$ term. (The more similar they are, the higher the inner product will be) which is a $T\times T$ "weight" matrix.

Each row of the output of our attention Matrix is a linear combination of all the rows of V, in other words, this is now mixing together elements of different points in time.

Properties of self-attention:

- Invariant (really, equivariant)  to permutations of the $K,Q,V$ matrices

- Allows influence between $k_t,q_t,v_t$ over *all* times, without increasing the parameter count.

- Compute cost is $O(T^2+Td)$ (cannot be easily reduced due to nonlinearity applied to full $T\times T$ matrix)

self-attention mixes together the entire sequence no matter how long that sequence is it provides kind of a single layer mixing of the entire sequence without any additional parameters. 

self-attention has no parameters, it is just a non-linearity that's able to mix over all time with no favoritism for the current time versus previous time because it's permutationally invariant and potentially equivalent  

### Transformers for time series

The transformer architecture uses a series of attention mechanisms (and feed  forward layers) to process a time series
$$
Z^{(i+1)}=\text{Transformer}(Z^{(i)})
$$
All time steps (in practice, within a given time slice are processed in parallel, avoids the need for sequential processing as in RNNs)


![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.1ap030nj8b.webp)

Transformer takes a series of inputs and transform them to a series of hidden states up until we finally get our output using this mechanism we call the Transformer which is in turn is going to be a particular construct of an architecture which uses the self-attention layer. 

### Transformer block

In more detail, the Transformer block has the following form:

$$\begin{align}
\widetilde{Z_1}&=\text{selfAttention}(Z^{(i)}W_k, Z^{(i)}W_Q,Z^{(i)}W_V)\\
&= \text{softmax}(\frac{Z^{(i)}W_kW_Q^TZ^{(i)T}}{\sqrt{d}})Z^{(i)}W_V
\end{align}$$
$$
\widetilde{Z_2}=\text{LayerNorm}(Z^{(i)}+\widetilde{Z_1})
$$
$$
Z^{(i+1)} = \text{LayerNorm}(\widetilde{Z_2}+\text{ReLU}(\widetilde{Z_2}W_1)W_2)
$$

what it really is just a self-attention layer which mixes together the different components on time. 

**Pros**: 

- Full receptive field within a single layer (i.e., can immediately use past data)

- Mixing over time doesn't increase parameter count (unlike convolutions)

**Cons**:

- All outputs depend on all inputs (no good e.g., for autoregressive tasks)

- No ordering of data (remember that transformers are equivariant to permutations of the sequence)

### Masked self-attention

To solve the problem of "acausal" dependencies, we can *mask* the softmax operator to assign zero weight to any "future" time steps
$$
\text{softmax}(\frac{KQ^T}{\sqrt{d}}-M)V
$$
Note that even though technically this means we can "avoid" creating those entries in the attention matrix to being with, in practice it's often faster to just form them then mask them out.

### Positional encodings

To solve the problem of "order invariance", we can add a **positional encoding** to the input, which associates each input with its position in the sequence.

$$
X\in R^n=\left[
\begin{matrix}
——x_1^T——\\
——x_2^T——\\
\vdots\\
——x_T^T——
\end{matrix}
\right]
+
\left[\begin{matrix}
\sin(w_1*1)&\sin(w_2*1)&\cdots&\sin(w_n*1)\\
\sin(w_1*2)&\sin(w_2*2)&\cdots&\sin(w_n*2)\\
\vdots&\vdots&\cdots&\vdots\\
\sin(w_1*T)&\sin(w_2*T)&\cdots&\sin(w_n*T)
\end{matrix}\right]
$$

## Transformers beyond time series 

Recent work has observed that transformer blocks are extremely powerful beyond just time series

- Vision Transformers: Apply transformer to image (represented by a collection of patch embeddings), works better than CNNs for large data sets

- Graph Transformers: Capture graph structure in the attention matrix

In all cases, some challenges are:

- How to represent data such that $O(T^2)$ operations are feasible 

- How to form positional embeddings

- How to form the mask matrix

## Implementation

```python
import numpy as np
import torch
import torch.nn as nn

def softmax(Z):
	Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
	return Z / Z.sum(axis=-1, keepdims=True)

def self_attention(X, mask, W_KQV, W_out):
	K,Q,V = np.split(X@W_KQV, 3, axis=1)
	attn = softmax(K@Q.T / np.sqrt(X.shape[1]) + mask)
	return attn @ V @ W_out, attn

T, d = 100, 64
attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
M = torch.triu(-float("inf")*torch.ones(T, T), 1)
X = torch.randn(1, T, d)
Y_, A_ = attn(X, X, X, attn_mask=M)

Y, A = self_attention(X[0], numpy(), M.numpy(),
					 attn.in_proj_weight.detach().numpy().T,
					 attn.out_proj.weight.detach().numpy().T)
```

### Mini batching

$K\in\mathbb{R}^{T\times d}$

$T\times B\times d$ for RNNs

$B\times T\times d$ for Transformers

Batch Matrix Multiplication (BMM)

```python
c = np.random
```


### Multihead attention

```python
def multihead_attention(X, mask, heads, W_KQV, W_out):
	B, T, d = X.shape
	K, Q, V = np.split(X@W_KQV, 3, axis=-1)
	# B x T x d => B x heads x T x d/heads 
	K, Q, V = [a.reshape(B, T, heads, d // heads).swapaxes(1, 2) for a in (K, Q, V)]
	attn = softmax(K@Q.swapaxes(-1, -2) / np.sqrt(d // heads) + mask)
	return (attn @ V).swapaxes(1, 2).reshape(B, T, d) @ W_out, attn
	
```

```python
heads = 4
attn = nn.MultiheadAttention(d, heads, bias=Fasle, batch_first=True)
Y_, A_ = attn(X, X, X, attn_mask=M)
```

```python
Y, A = multihead_attention(X.numpy(), M.numpy(), heads,
							attn.in_proj_weight.detach().numpy().T,
							attn.out_proj.weight.detach().numpy().T)
```

### Transformer Block



```python
def layer_norm(Z, eps):
	return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1,keepdims=True) + eps)

def relu(Z):
	return np.maximum(Z, 0)

def transformer(X, mask, heads, W_KQV, W_out, W_ffl, W_ff2, eps):
	Z = layer_norm(X + multihead_attention(X, mask, heads, W_KQV, W_out)[0], eps)
	return layer_norm(Z + relu(Z@W_ff1)@W_ff2 + eps)

```

```python
trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128,
								   dropout=0.0, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
Y_ = trans(X, M)
```

```python
Y = transformer(X.numpy(), M.numpy(), heads, trans.self_attn.in_proj_weight.detach().numpy().T, 
trans.self_attn.out_proj_weight.detach().numpy().T,
trans.self.linear1.weight.detach().numpy().T,
trans.self.linear2.weight.detach().numpy().T,
trans.norm1.eps)
```