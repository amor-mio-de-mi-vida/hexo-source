---
date: 2024-10-28 18:37:20
date modified: 2024-10-29 13:59:55
title: Sequence Modeling and Recurrent Networks
tags:
  - course
categories:
  - DLSystem
---
## Sequence Modeling

In the samples we have considered so far, we make predictions assuming each input output pair $x^{(i)}, y^{(i)}$ is. independent identically distributed.

In practice, many cases where the input/output pairs are given in a specific *sequence*, and we need to use the information about this sequence to help us make predictions.

## Recurrent neural networks

Recurrent neural networks (RNNs) maintain a hidden state over time, which is a function of the current input and previous hidden state.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.6t74g9v60e.webp)

$$\begin{align}
&h_t=f(W_{hh}h_{t-1}+W_{hx}x_t+b_h)\\
&y_t=g(W_{yh}h_t+b_y)\\
&h_t\in \mathbb{R}^d,x_t\in\mathbb{R}^n,y_t\in\mathbb{R}^k\\
&W_{hh}\in\mathbb{R}^{d\times d},W_{hx}\in\mathbb{R}^{d\times n},b\in\mathbb{R}^d\\
&W_{yh}\in\mathbb{R}^{k\times d},b_y\in\mathbb{R}^{k}
\end{align}$$

Because our hidden state depends on the previous hidden state, in some sense, the hidden state at time t captures or can capture, in the ideal case, all the inputs it has seen up until that point.

```pseudo algorithm 
opt = Optimizer(params= (W_hh, W_hx, W_yh, b_h, b_y))
h[0] = 0
l = 0
for t = 1, ..., T:
	h[t] = f(W_hh @ h[t-1] + W_hx @ x[t] + b_h)
	y[t] = g(W_yh @ h[t] + b_y)
	l += Loss(y[t], y_star[t])
l.backward()
opt.step()
```

### Stacking RNNs

Just like normal neural networks, RNNs can be stacked together, treating the hidden unit of one layer as the input to the next layer, to form "deep" RNNs.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.51e5ldcdk6.webp)

### Exploding activations/ gradients

The challenge for training RNNs is similar to that of training deep MLP networks.

Because we train RNN on long sequences, if the weights/activation of the RNN are scaled poorly, the hidden activations (and therefore also the gradients) will grow unboundedly with sequence length.

Single layer RNN with ReLU activations, using weight initialization 
$$
W_{hh}\sim\mathcal{N}(0, 3/n)
$$
Recall that $\sigma^2=2/n$ was the "proper" initialization for ReLU activations.

### Vanishing activation/gradients

Similarly, if weights are too small then information from the inputs will quickly decay with time (and it is precisely the "long range" dependencies that we would often like to model with sequence models)

Single layer RNN with ReLU activations, using weight initialization
$$
W_{hh}\sim\mathcal{N}(0, 1.5/n)
$$

Non-zero input only provided here for time 1, showing decay of information about this input over time.

### Alternative Activations

One obvious problem with the ReLU is that it can grow unboundedly; does using bounded activations "fix" this problem?

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.51e5ldsoaf.webp)

No ... creating large enough weights to not cause activations/gradients to vanish requires being in the "saturating" regions of the activations, where gradients are very small $\Rightarrow$ still have vanishing gradients.

either activations be zero or gradient be zero.

## LSTM

Long short term memory (LSTM) cells are a particular form of hidden unit update that avoids (some of) the problems of vanilla RNNs

Step 1: Divide the hidden unit into two components, called (confusingly) the *hidden state* and the *cell* state.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.9nzsm2xptc.webp)

Step 2: Use a very specific formula to update the hidden state and cell state (throwing in some other names, like "forget gate", "input gate", "output gate" for good measure).

$$\begin{align}
&
\left[
\begin{matrix}
i_t\\
f_t\\
g_t\\
o_t
\end{matrix}
\right]
=
\left(
\begin{matrix}
sigmoid\\
sigmoid\\
tanh\\
sigmoid
\end{matrix}
\right)
(W_{hh}h_{t-1}+W_{hx}x_t+b_h)
\\
&c_t = c_{t-1}\odot f_t + i_t\odot g_t\\
&h_t = tanh(c_t)\odot o_t\\
&W_{hh}\in\mathbb{R}^{4d\times d},h_{t-1}\in\mathbb{R}^{d},
W_{hx}\in\mathbb{R}^{4d\times n},x_t\in\mathbb{R}^{n},
b_h\in\mathbb{R}^{4d}
\end{align}$$

### Why do LSTMs work?

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.5xan0uo96c.webp)

## Beyond "simple" sequential models

### Sequence-to-sequence models

Can concatenate two RNNs together, one that "only" processes the sequence to create a final hidden state (i.e. no loss function); then a section that takes in this initial hidden state, and "only" generates a sequence.

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.9dcysyhv50.webp)

### Bidirectional RNNs

RNNs can use *only* the sequence information up until time *t* to predict *y_t*.

- This is sometimes desirable (e.g., autoregressive models)

- But sometime undesirable (e.g., language translation where we want to use "whole" input sequence)

Bi-directional RNNs: stack a forward-running RNN with a backward-running RNN: information from the entire sequence to propagates to the hidden state.

## Implementation

LSTM for each element in the input sequence, each layer computes the following function:

$$\begin{align}
&i_t = \sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi})\\
&f_t = \sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})\\
&g_t = \text{tanh}(W_{ig}x_t + b_{ig} + W_{hg}h_{t-1}+b_{hg})\\
&o_t = \sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})\\
&c_t = f_t\odot c_{t-1}+i_t\odot g_t\\
&h_t = o_t\odot \text{tanh}(c_t)
\end{align}$$

### Single LSTM Cell

```python
model = nn.LSTMCell(input_size=20, hidden_size=100)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def lstm_cell(x, h, c, W_hh, W_ih, b):
	i, f, g, o = np.split(W_hh@h + W_ih@x + b, 4)
	i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
	c_out = f * c + i * g
	h_out = o * np.tanh(c_out)
	return h_out, c_out
```

### Full sequence LSTM

```python
model = nn.LSTM(input_size=20, hidden_size=100, num_layers=1)

def lstm(X, h, c, W_hh, W_ih, b):
	H = np.zeros((X.shape[0], h.shape[0]))
	for t in range(X.shape[0]):
		h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
		H[t] = h
	return H, c
```

```python
H, cn = lstm(X, h0[0], c0[0],
			model.weight_hh_10.detach().numpy(),
			model.weight_ih_10.detach().numpy(),
			(model.bias_hh_10 + model.bias_ih_10).detach().numpy())
			
H_, (hn_, cn_) = model(torch.tensor[X][:,None,:],
					   (torch.tensor(h0)[:,None,:], torch.tensor(c0)[:,None,:]))
```

### Batching

multiplication works best when matrices are contiguous together in memory.

`X[B, T, n]` $\Rightarrow$ `X[T, B, n]`

```python
def lstm_cell(x, h, c, W_hh, W_ih, b):
	i, f, g, o = np.split(h@W_hh + x@W_ih@ + b, 4, axis=1)
	i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
	c_out = f * c + i * g
	h_out = o * np.tanh(c_out)
	return h_out, c_out

def lstm(X, h, c, W_hh, W_ih, b):
	H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
	for t in range(X.shape[0]):
		h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
		H[t] = h
	return H, c
```

```python
H, cn = lstm(X, h0[0], c0[0],
			model.weight_hh_10.detach().numpy().T,
			model.weight_ih_10.detach().numpy().T,
			(model.bias_hh_10 + model.bias_ih_10).detach().numpy())
			
H_, (hn_, cn_) = model(torch.tensor[X],
					   (torch.tensor(h0), torch.tensor(c0)))
```

### Training LSTMs

```python
opt = optim.SGD([W_hh, W_ih, b])

def train_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
	H, cn = lstm(X, h0, c0, W_hh, W_ih, b)
	l = loss(H, Y)
	l.backward()
	opt.step()

def tran_deep_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
	H = X
	depth = len(W_hh)
	for d in range(depth):
		H, cn = lstm(H, h0[d], c0[d], W_hh[d], W_ih[d], b[d])
		l = loss(H, Y)
		l.backward()
		opt.step()

# large sequence -> split into blocks
def tran_deep_lstm_by_block(X, h0, c0, Y, W_hh, W_ih, b, opt):
	H = X
	depth = len(W_hh)
	for d in range(depth):
		H, cn = lstm(H, h0[d], c0[d], W_hh[d], W_ih[d], b[d])
		ho[i] = H[-1].detach().copy()
		c0[d] = cn.detach().copy()
		l = loss(H, Y)
		l.backward()
		opt.step()
	return h0, c0


h0, c0 = zeros(...)
for i in range(seqence_len // block_size):
	h0, c0 = train_deep_lstm_by_block(X[i*block_size:(i+1)*block_size], h0, c0, Y[i*block_size:(i+1)*block_size], W_hh, W_ih, b, opt)
```