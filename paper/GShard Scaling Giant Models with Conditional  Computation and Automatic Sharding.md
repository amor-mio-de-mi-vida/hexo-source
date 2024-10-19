---
date: 2024-10-17 22:29:32
date modified: 2024-10-19 21:21:25
title: "GShard: Scaling Giant Models with Conditional\r Computation and Automatic Sharding"
tags:
  - paper
categories:
  - " paper"
---

摘要：

神经网络扩展对于提高许多现实世界机器学习应用中的模型质量至关重要，这些应用拥有大量的训练数据和计算资源。尽管这种扩展趋势被证实是提高模型质量的可靠方法，但在实现过程中仍存在挑战，如计算成本、编程的便利性以及在并行设备上的高效实现。<mark>GShard是一个由一组轻量级注释API和一个对XLA编译器的扩展组成的模块。它提供了一种优雅的方式来表达广泛的并行计算模式，只需对现有模型代码进行最小的更改。</mark>GShard使我们能够使用<mark>自动分片</mark>将多语言神经机器翻译Transformer模型与稀疏门控混合专家（Sparsely-Gated Mixture-of-Experts）扩展到超过6000亿参数。我们展示了一个如此巨大的模型可以在4天内使用2048个TPU v3加速器进行高效训练，从而在将100种语言翻译成英语的质量上，比现有技术取得了显著提高。

问题：

- 现有的模型框架对高效的模型并行方法支持不足。

- 计算成本相对于模型大小超线性增长。对于在多个设备上分割层权重和计算来实现模型并行的方法，由于底层神经网络负载不均衡和顺序依赖性，导致分布式训练效率低下。

- 对于大规模模型计算图构建和编译时间过长。

- 将模型有效的划分到多个设备上运行是具有挑战性的。对于图级划分，需要复杂的算法来减少由不同设备上分配的图的不同部分之间的顺序依赖性引入的开销。对于操作符级并行，不同的划分操作符有不同的通信模式。

解决方法

模型背景 (模型角度优化)

MoE 层由 E 个前馈神经网络构成，通过Gate() 函数来选择：
$$\begin{align}
&\varsigma_{s, E}=GATE(x_s)\\
& FFN_e(x_s)=w_{Oe}\cdot\text{ReLU}(w_{i_e}\cdot x_s)\\
& y_s=\overset{E}{\underset{e=1}{\sum}}\varsigma_{s,e}\cdot FFN_e(x_s)
\end{align}$$


![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.2veqlsv4pt.webp)

模型并行体现在把 MoE 层分布到不同的设备上， 每个设备负责计算不同 expert 的部分。网络的其它层通过复制得到。同时选择让每个 token 最多分配给两个 expert。

门控函数 gating 对MoE层至关重要，该层由 softmax 激活函数建模，以指示每个 expert 在处理传入 token 时的权重。为了高效实现，gating层需要满足以下两个目标：

- Balanced load：训练期间看到的大多数 token 都会被分派给少数 expert，为少数（忙碌的） expert 积累了非常大的输入缓冲区，而其他专家没有经过训练，从而减慢了训练速度，其他 expert 也没有得到足够的训练。

- Efficiency at scale：门控函数的顺序实现将使大部分计算资源在大部分时间处于空闲状态。

通过以下技术实现

- Expert capacity： 为了确保负载平衡，我们强制要求一个专家处理的代币数量低于某个统一阈值。$\text{GATE}()$保持一个正在运行的计数器 $c_e$，用于调度给 expert 的代币数量。当 token 选择的两个 expert 都已经超过其容量时，该 token 被视为溢出令牌，$\varsigma_{s,E}$ 退化成零向量。此类 token 通过残差连接将其表示传递到下一层。

- Local group dispatching：将训练批次中的所有 token 均匀划分为 G 组，即每组包含 $S = N/G$ 个 token。所有组都独立并行处理。每个组都被赋予每个专家的分数能力 $2N/(G\cdot E)$ 。每个组确保最多将这么多令牌分派给专家。通过这种方式，我们可以确保专家容量仍然得到实施，并且总体负载是平衡的。

- Auxiliary loss: 门控功能并不总是选择相同的几个 EA，因为这会导致只有少数 EA 的容量溢出，而其余 EA 的利用率不足。我们定义辅助损失函数$\mathscr{l}_{aux}$，他被添加到模型整体的损失函数$\mathcal{L}=\mathscr{l}_{nll}+k*\mathscr{l}_{aux}$ 中，，具有常数乘数 $k$。$c_e/S$ 代表了输入到各个 expert 的分数，我们想要最小化其均方误差。但是由于$c_e$ 是从 top-2 运算中得出的，不可微分，因此我们使用每个专家$m_e$ 的平均分数作为可微分近似值，并将 $(c_e/S)^2$ 换为 $m_e(c_e/S)$，可以用梯度下降方法优化。

- Random routing: 直观地说，因为 $y_s$ 是所选 expert 返回的加权平均值，所以如果第 2 个 expert 的权重非常小，我们可以简单地忽略第 2 个 expert 以保存整体 expert 能力。因此，除了遵守 expert 能力约束外，$\text{GATE}()$ 还调度给概率与其权重 $g_2$ 成正比的第二好的 expert。

使用 GShard 的高度并行实现

MoE 层的表达 

gating函数算法
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.8dwv20ecbd.webp)
MoE层算法
![image](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.2yycjl03xc.webp)

用于并行执行的 GShardAnnotation API

GShard 中的分片 API 允许我们在程序中注释张量，以选择性地指定应该如何对它们进行分区。此信息将传播到编译器，以便编译器可以自动应用转换以进行并行执行。

主要应用的API：

- `replicate(tensor)` : 注释要跨分区复制的张量，并返回带注释的张量。这通常用于模型中的非 MoE 层来复制权重。

- `split(tensor, split_dimension, num_partitions)`: 注释要沿 `split_dimension` 分区的 Tensor，并返回带注释的 Tensor。分区 i 放置在第 i 台设备上，`num_partitions`不得超过系统上的设备数。

- `shard(tensor， device_assignment)`: 将 `split()` 通用化，以允许对多个维度进行分区并指定每个分区的位置。


SPMD 架构

本节介绍基于分片注释自动对计算图进行分区的编译器基础设施。分片注释通知编译器每个张量应如何在设备之间分布。SPMD （Single Program Multiple Data） 分区器（或为简单起见的“partitioner”）是一个编译器组件，它将计算图转换为单个程序，以便在所有设备上并行执行。这使得编译时间几乎是恒定的，无论分区数量如何，这使我们能够扩展到数千个分区。

