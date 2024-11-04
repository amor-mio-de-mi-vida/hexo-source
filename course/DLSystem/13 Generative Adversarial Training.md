---
date: 2024-11-02 16:56:03
date modified: 2024-11-02 20:08:27
title: Generative Adversarial Training
tags:
  - course
categories:
  - DLSystem
---
## Generative Adversarial Training

### From classifier to generator


### Learn generator through an oracle discriminator

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.8vmxajoyja.webp)
Assume that we have an oracle discriminator that can tell the difference between real and fake data. Then we need train the generator to "fool" the oracle discriminator. We need to maximize the discriminator loss

Generator objective: $max_G\{-E_{z\sim Noise}log(1-D(G(z)))\}$

### Learning the discriminator 

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.5mntdw4yy6.webp)

We do not have an oracle discriminator, but we can learn it using the real and generated fake data.

Discriminator objective $min_D\{-E_{x\sim Data}logD(x)-E_{z\sim Noise}log(1-D(G(z)))\}$

### Generative adversarial training in practice

Iterative process

**Discriminator update**

- Sample minibatch of $D(G(z))$, get a minibatch of $D(x)$

- Update *D* to minimize $min_D\{-E_{x\sim Data}logD(x)-E_{z\sim Noise}log(1-D(G(z))\}$

**Generator update**

- Sample minibatch of $D(G(z))$

- Update*G* to minimize $max_G\{-E_{z\sim Noise}log(1-D(G(z)))\}$ , this can be done by feeding $\text{label} = 1$ to the model

## Adversarial training as a module in deep learning models

### Use GAN as a compositional module

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.8ad9oa6y4v.webp)

GAN is not exactly like a loss function, as it involves an iterative update recipe. But we can compose it with other neural network modules in a similar way like loss function. 

Use GAN "loss" whenever we want a collection of data to "look like" another collection.

### DCGAN: Deep convolutional generative adversarial networks

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.1e8m43rv6t.webp)

How can this modularization help?

### CycleGAN

![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/dlsystem/image.39l6wqf3wb.webp)

What are other ways to compose GAN module together with other deep learning components? 

## Implementation

### Prepare the training dataset

For demonstration purpose, we create out "real" dataset as a two dimensional gaussian distribution.

$$X\sim \mathcal{N}(\mu, \Sigma), \Sigma = A^TA$$

```python
A = np.array([[1, 2], [-0.2, 0.5]])
mu = np.array([2, 1])
# total number of sample data to generated
num_sample = 3200
data = np.random.normal(0, 1, (num_sample, 2)) @ A + mu
```
### Generator network G

Now we are ready to build our generator network G, to keep things simple, we make generator an one layer linear neural network.

```python
model_G = nn.Linear(2, 2)

def sample_G(model_G, num_samples):
	Z = ndl.Tensor(np.random.normal(0, 1, (num_sample, 2)))
	return models_G(Z).numpy()
```

### Discriminator D

```python
model_D = nn.Sequential(
	nn.Linear(2, 20),
	nn.ReLU(),
	nn.Linear(20, 10),
	nn.ReLU(),
	nn.Linear(10, 2)
)
loss_D = nn.SoftmaxLoss()
```

### Generative adversarial training

A Generative adversarial training process iteratively update the generator G and discriminator D to play a "minimax" game.

$$\underset{D}{\min}\underset{G}{\max}\{-E_{x\sim Data}\log D(x)-E_{z\sim Noise}\log(1-D(G(z))\}$$

Note that however, in practice, the G update step usually use an alternative objective function.

$$\underset{G}{\min}\{-E_{z\sim Noise}\log D(G(z))\}$$

### Generator update

```python
opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)

def update_G(Z, model_G, model_D, loss_D, opt_G):
	X_fake = model_G(Z)
	Y_fake = model_D(X_fake)
	batch_size = Z.shape[0]
	ones = ndl.ones(num_samples)
	loss = loss_D(Y_fake, ones)
	loss.backward()
	opt_G.step()
```

### Discriminator update

```python
opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)

def update_D(Z, X, model_G, model_D, loss_D, opt_D):
	X_fake = model_G(Z).detach()
	Y_fake = model_D(X_fake)
	Y_real = model_D(X)
	batch_size = Z.shap[0]
	ones = ndl.ones(batch_size, dtype="int32")
	zeros = ndl.zeros(batch_size, dtype="int32")
	
	loss = loss_D(Y_fake, zeros) + loss_D(Y_real, ones)
	loss.backward()
	opt_D.step()
```

### Put it together

```python
def train_gan(data, batch_size, num_epochs):
	assert data.shape[0] % batch_size == 0

	for epoch in range(num_epochs):
		begin = (batch_size * epoch) % data.shape[0]
		X = ndl.Tensor(data[begin: begin + batch_size])
		Z = ndl.Tensor(np.random.normal(0, 1, (batch_size, 2)))

		update_G(Z, model_G, model_D, loss_D, opt_G)
		update_D(X, Z, model_G, model_D, loss_D, opt_D)

train_gan(data, batch_size, 2000)
```

### Modularizing GAN "Loss"

```python

class GANLoss:
	def __init__(self,model_D, opt_D):
		self.model_D = model.D
		self.opt_D = opt_D
		self.loss_D = nn.SoftmaxLoss()
		
	def _update_D(self, X_fake, X_real):
		Y_fake = self.model_D(X_fake.detach())
		Y_real = self.model_D(X_real)
		batch_size = Y_fake.shape[0]
		ones = ndl.ones(batch_size, dtype="int32")
		zeros = ndl.zeros(batch_size, dtype="int32")
		loss = self.loss_D(Y_fake, zeros) + self.loss_D(Y_real, ones)
		loss.backward()
		self.opt_D.step()
		
	def forward(self, X_fake, X_real):
		self._update_D(X_fake, X_real)
		Y_fake = self.model_D(X_fake)
		batch_size = Y_fake.shape[0]
		ones = ndl.ones(batch_size, dtype="int32")
		loss = self.loss_D(Y_fake, ones)
		return loss
		
```

```python
model_G = nn.Sequential(nn.Linear(2, 2))
opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)

model_D = nn.Sequential(
	nn.Linear(2, 20),
	nn.ReLU(),
	nn.Linear(20, 10),
	nn.ReLU(),
	nn.Linear(10, 2)
)
opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)
gan_loss = GANLoss(model_D, opt_D)

def train_gan(data, batch_size, num_epochs):
	assert data.shape[0] % batch_size == 0
	for epoch in range(num_epochs):
		opt_G.reset_grad()
		begin = (batch_size * epoch) % data.shape[0]
		X = data[begin: begin+batch_size, :]
		Z = np.random.normal(0, 1, (batch_size, 2))
		X = ndl.Tensor(X)
		Z = ndl.Tensor(Z)
		fake_X = model_G(Z)
		loss = gan_loss.forward(fake_X, X)
		loss.backward()
		opt_G.step()

train_gan(data, 32, 2000)

```