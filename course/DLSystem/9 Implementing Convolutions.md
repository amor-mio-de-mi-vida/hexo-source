---
date: 2024-10-28 10:05:59
date modified: 2024-10-28 14:50:32
title: Implementing Convolutions
tags:
  - course
categories:
  - DLSystem
date created: 2024-09-25 13:26:08
---
### Storage Order

We could represent a hidden unit as an array 

`float Z[BATCHES][HEIGHT][WEIGHT][CHANNELS];`

as NHWC format (number(batch)-height-width-channel). PyTorch defaults to NCHW format.

we will store the weights in the form:

`float weights[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][OUT_CHANNELS];`

PyTorch defaults `OUT_CHANNELS x IN_CHANNELS x KERNEL_SIZE x KERNEL_SIZE`.

### Convolutions with simple loops

baseline
```python
import torch
import torch.nn as nn

def conv_reference(Z, weight):
	# NHWC -> NCHW
	Z_torch = torch.tensor(Z).permute(0, 3, 1, 2)
	
	# KKIO -> OIKK
	W_torch = torch.tensor(weight).permute(3, 2, 0, 1)
	
	# run convolution
	out = nn.functional.conv2d(Z_torch, W_torch)
	
	# NCHW -> NHWC
	return out.permute(0, 2, 3, 1).contiguous().numpy()
```
naive convolution
```python
def conv_naive(Z, weight):
	N,H,W,C_in = Z.shape
	K,_,_,C_out = weight.shape
	
	out = np.zeros((N, H-K+1, W-K+1,C_out))
	for n in range(N):
		for c_in in range(C_in):
			for c_out in range(C_out):
				for y in range(H-K+1):
					for x in range(W-K+1):
						for i in range(K):
							for j in range(K):
								out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i, j, c_in, c_out]
	return out
```

###  Convolutions as matrix multiplications
```python
def conv_matrix_mult(Z, weight):
	N,H,W,C_in = Z.shape
	K,_,_,C_out = weight.shape
	
	for i in range(K):
		for j in range(K):
			out += Z[:, i:i+H-K+1, j:j+W-K+1, :] @ weight[i, j]
	return out
```

### Manipulating matrices via strides

Normally we think of storing a matrix as a 2D array

`float A[M][N];`

in order to make better use of the caches and vector operations in  modern CPUs, it was beneficial to lay our matrix memory groups by individual small "tiles", so that the CPU vector operations could efficiently access operators

`float A[M/TILE][N/TILE][TILE][TILE]`

where `TILE` is some small constant (like 4), which allows the CPU to use its vector processor to perform very efficient operations on `TILE x TILE` blocks.

```python
import numpy as np
n = 6
A = np.arange(n**2, dtype=np.float32).reshape(n, n)
print(A)
```

```python
import ctypes
print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
```

```python
B = np.lib.stride_tricks.as_strided(A, shape=(3,3,2,2), strides=np.array((12,2,6,1))*4)
print(B)
```

```python
print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=B.nbytes), B.dtype, B.size))
```

```python
C = np.ascontiguousarray(B)
print(np.frombuffer(ctypes.string_at(C.ctypes.data, size=C.nbytes), C.dtype, C.size))
```

### Convolutions via im2col

Essentially, we want to bundle all the computation needed for convolution into a single matrix multiplication, which will then leverage all the optimizations that we can implement for normal matrix multiplication.

The key approach to doing this is called the `im2col` operator, which "unfolds" a 4D array into exactly the form needed to perform multiplication via convolution.

The key will be to form a $(H-K+1)\times (W-K+1)\times K\times K$ array, then flatten it to a matrix we can multiply by the filter.

> There is a very crucial point to make regarding memory efficiency of this operation. While reshaping `W` into an array (or what will be a matrix for multi-channel convolutions) is "free", in that it does't allocate any new memory, reshaping the `B` matrix above is very much *not* a free operation. Specifically, while the strided form of `B` uses the same memory as `A`, once we actually convert `B` into a 2D matrix, there is no way to represent this data using any kind of strides, and we have to just allocate the entire matrix. This means we actually need to *form* the full im2col matrix, which requires $O(K^2)$ more memory than the original image, which can be quite costly for large kernel sizes.
> 
> For this reason, in practice it's often the case that the best modern implementations *won't* actually initiate the the full  im2col matrix, and will instead perform a kind of "lazy" formation, or specialize the matrix operation natively to im2col matrices in their native strided form.

Instead of forming a 4D $(H-K+1)\times (W-K+1)\times K\times K$ array, we form a 6D $N\times (H-K+1)\times (W-K+1)\times K\times K$ array. We can apply the same trick by just repeating the strides for dimensions 1 and 2 (the height and width) for dimensions 3 and 4 (the $K\times K$ blocks), and leave the strides for the minibatch and channels unchanged. You can just use the strides of the $Z$ input and repeat whatever they are.

To compute the convolution, flatten the im2col matrix to a $(N\cdot(H-K+1)\cdot(W-K+1))\times(K\cdot K\cdot C)$ matrix (remember this operation is highly memory inefficient), flatten the weights array to a $(K\cdot K\cdot C)\times C_{out}$ matrix, perform the multiplication, and resize back to the desired size of the final 4D array output.  

```python
def conv_im2col(Z, weight):
	N,H,W,C_in = Z.shape
	K,_,_,C_out = weight.shape
	Ns, Hs, Ws, Cs = Z.strides
	
	inner_dim = K * K * C_in
	A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in), strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1, inner_dim)
	out = A @ weight.reshape(-1, C_out)
	return out.reshape(N, H-K+1, W-K+1, C_out)
```