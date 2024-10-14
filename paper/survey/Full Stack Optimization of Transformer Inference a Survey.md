---
title: "Full Stack Optimization of Transformer Inference: a Survey"
date: 2024-10-13 21:22:09
tags:
  - Hexo
  - Keep
categories:
  - Hexo
date modified: 2024-10-14 12:37:02
---
最近Transformer模型在进行推理时所需的计算量和带宽正在以显著的速度增长，这使得它们在延迟敏感型应用中的部署变得具有挑战性。因此，人们越来越关注如何提高Transformer模型的效率，方法从改变架构设计，到开发专门的领域特定加速器不等。在这项工作中我们调研了不同的高效Transformer推理方法，包括：

(i) 分析和剖析现有Transformer架构中的瓶颈，以及它们与之前的卷积模型在相似性和差异性；

(ii) Transformer架构对硬件的影响，包括非线性运算如层归一化（Layer Normalization）、Softmax和GELU，以及线性运算对硬件设计的影响；

(iii) 优化固定Transformer架构的方法；

(iv) 在为Transformer模型寻找正确的操作映射和调度方面所面临的挑战；以及(v) 通过使用神经架构搜索来适应架构，优化Transformer模型的方法。

最后，我们通过对Gemmini（一个开源的全栈深度神经网络加速器生成器）应用所调研的优化方法进行了一个案例研究，并展示了每种方法相比于之前在Gemmini上的基准测试结果所带来的改进。

<!-- more-->

