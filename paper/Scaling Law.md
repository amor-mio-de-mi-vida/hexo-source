---
date: 2024-11-27 16:26:26
date modified: 2024-11-27 16:26:41
title: Scaling Law
tags: 
categories: 
date created: 2024-09-25 13:26:08
---

## Scaling Laws for Neural Language Models

**Time: 23 Jan 2020**

**Abstract:**

我们研究语言模型在交叉熵损失上的经验标度律。 **损失随着模型大小、数据集大小以及用于训练的计算量按幂律变化**，其中一些趋势跨越了七个数量级以上。**其他架构细节，如网络宽度或深度，在很大范围内影响微乎其微**。简单的方程式控制着过拟合与模型/数据集大小之间的关系，以及训练速度与模型大小之间的关系。这些关系使我们能够确定固定计算预算的最优分配。更大的模型在样本效率上显著更高，因此最优计算效率的训练涉及在相对较少的数据上训练非常大的模型，并在收敛前显著停止。

**Take away:**

- **Performance depends strongly on scale, weakly on model shape**: Model performance depends most strongly on scale, which consists of three factors: the number of model parameters $N$ (excluding embeddings), the size of the dataset $D$, and the amount of compute $C$ used for training. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width.

- **Smooth power laws**: Performance has a power-law relationship with each of the three scale factors $N$, $D$, $C$ when not bottlenecked by the other two, with trends spanning more than six orders of magnitude. We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss.

- **Universality of overfitting**: Performance improves predictably as long as we scale up $N$ and $D$ in tandem, but enters a regime of diminishing returns if either $N$ or $D$ is held fixed while the other increases. The performance penalty depends predictably on the ratio $N^{0.74}/D$, meaning that every time we increase the model size 8x, we only need to increase the data by roughly 5x to avoid a penalty.

- **Universality of training**: Training curves follow predictable power-laws whose parameters are roughly independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the loss that would be achieved if we trained for much longer. 

- **Transfer improves with test performance**: When we evaluate models on text with a different distribution than they were trained on, the results are strongly correlated to those on the training validation set with a roughly constant offset in the loss– in other words, transfer to a different distribution incurs a constant penalty but otherwise improves roughly in line with performance on the training set.

- **Sample efficiency**: Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps and using fewer data points

- **Convergence is inefficient**: When working within a fixed compute budget $C$ but without any other restrictions on the model size $N$ or available data $D$, we attain optimal performance by training very large models and stopping significantly short of convergence. Maximally compute-efficient training would therefore be far more sample efficient than one might expect based on training small models to convergence, with data requirements growing very slowly as $D\sim C^{0.27}$ with training compute.

- **Optimal batch size**: The ideal batch size for training these models is roughly a power of the loss only, and continues to be determinable by measuring the gradient noise scale; it is roughly 1-2 million tokens at convergence for the largest models we can train.

Taken together, these results show that language modeling performance improves smoothly and predictably as we appropriately scale up model size, data, and compute. We expect that larger language models will perform better and be more sample efficient than current models.


