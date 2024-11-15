---
date: 2024-11-11 21:28:15
date modified: 2024-11-11 21:28:56
title: Survey on Large Scale Neural Network Training
tags:
  - paper
  - survey
categories:
  - " paper"
---
## Abstract

现代深度神经网络（DNNs）需要大量的内存来存储权重、激活以及其他训练过程中的中间张量。因此，许多模型无法适应单个GPU设备，或者只能使用每个GPU上的小批量大小进行训练。本综述提供了一种系统性的概述，介绍了使DNNs训练更加高效的方法。我们分析了节省内存并充分利用具有单个或多个GPU架构上的计算和通信资源的技术。我们总结了策略的主要类别，并在类别内和类别间进行了比较。除了文献中提出的方法外，我们还讨论了可用的实现。

<!-- more -->