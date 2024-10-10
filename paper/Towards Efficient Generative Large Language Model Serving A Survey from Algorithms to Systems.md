---
date created: 2024-10-09 13:14:45
date modified: 2024-10-10 23:10:11
title: "Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems"
tags: 
categories: 
mathjax: "true"
date: 2024-10-10 09:07:32
---
## Abstract



## 1 Introduction

LLM 的空前成功也带来了一些挑战，最明显的是它们在服务期间的巨大计算要求。巨大的模型大小和复杂性，加上对大量计算资源的需求，阻碍了它们在实际应用中的广泛部署。这些模型的资源密集型性质引发了对能耗、可扩展性和可访问性的担忧，阻碍了它们在没有丰富计算资源的更广泛社区中的采用。
### Objectives
本调查的主要目的是全面概述 LLM 服务和推理的最新进展。我们将根据现有技术的基本方法对其进行系统回顾和分类，突出它们的优势和局限性。该调查将涵盖广泛的方法，包括解码算法、架构设计、模型压缩、低位量化、并行计算、内存管理、请求调度和内核优化。
### Structure
## 2 Background
### Transformer-based LLM

从数学上讲，Transformer 中的自注意力机制可以描述如下：对于输入序列
\$\$X=[x_1, x_2, ..., x_n]\$\$
Transformer 使用 $X$ 的线性变换计算一组查询 $Q$、键 $K$ 和 $valuesV$。然后，自我注意分数计算为：
\$\$\text{Attention}(Q, K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V\$\$
其中$d_k$是键的维度。这种机制允许模型专注于输出每个元素的 input 序列的不同部分，捕获复杂的依赖关系，而不管它们在 input sequence 中的距离如何。

Transformer 中的另一个重要结构是前馈网络 （FFN），它存在于 Transformer 的每一层中，并显着影响其计算强度。FFN 通常由两个线性变换组成，中间有一个非线性激活函数，通常表示为：
\$\$\text{FFN}(x)=max(0, xW_1+b_1)W_2+b_2\$\$
其中，$W_1$、$W_2$、$b_1$ 和 $b_2$ 是 FFN 的可学习参数，非线性函数 $max（0，·）$（在本例中为 ReLU）将必要的非线性引入模型，使其能够学习更复杂的模式。FFN 负责模型参数计数的很大一部分，因此负责其内存占用和计算负载。在每个 Transformer 层中，在多头注意力 （MHA） 聚合来自输入不同部分的信息后，FFN 会为每个位置独立处理这些聚合信息。这种并行处理能力是 Transformer 的一个关键优势，使其能够有效地处理序列。但是，这也意味着计算负载和内存需求随输入序列的长度和网络的深度而变化。

在基于 Transformer 的 LLM 中，自我注意和 FFN 的结合使这些模型能够捕获广泛的语言上下文和细微差别，从而在各种 NLP 任务中设定新的基准。然而，训练和推理的大量计算要求已成为一个关键的研究领域，专注于在不显著影响性能的情况下优化这些方面。Transformer 模型还包括其他关键组件，如位置编码，它添加了有关序列中每个标记位置的信息，以及多头注意力机制，它允许模型关注不同表示空间中序列的不同部分。
### GPUs and Other Accelerators
### LLM Inference

先前的研究对基于Transformer的LLM推理的算法强度进行了深入分析（例如，计算浮点运算次数、I/O和内存消耗），并根据自回归解码算法的执行提供了广泛的实证结果进行成本估算（例如，建模推理延迟[50]）。大型语言模型推理的优化是一个复杂的问题，因为可能存在不同的最优策略，不同的算法配置和系统设置。
### Challenges

- **延迟和响应时间**
	高效的大型语言模型推理需要实现低延迟和快速响应时间，尤其是在聊天机器人、虚拟助手和交互式系统等实时应用程序中。平衡模型复杂性和推理速度是一项关键挑战，需要优化算法和系统架构，以便在不影响准确性的情况下最大限度地减少响应时间。
- **内存占用和模型大小**
	大型语言模型由于其庞大的体积和包含的大量参数，对内存有着显著的需求。在内存受限的设备上部署这类模型是一个挑战，这要求开发有效的模型压缩技术和系统优化措施，以减少内存占用，同时不牺牲性能。
- **可扩展性和吞吐量**
	推理系统在生产环境中经常面临不同级别的请求负载。确保可扩展性和高吞吐量以有效地处理多个同时请求需要并行计算、请求调度和其他系统级优化，以便在资源之间有效地分配计算工作负载。
- **硬件兼容性和加速**
	有效地利用硬件资源对于大型语言模型推理至关重要。将大型语言模型适应于多样化的硬件平台和架构，包括中央处理器（CPUs）、图形处理器（GPUs）和专业加速器，需要硬件感知的算法设计和优化，以充分利用底层硬件的潜力。
-   **准确性与效率之间的权衡**
	优化大型语言模型（LLM）推理的效率有时可能涉及到与模型准确性的权衡。在模型大小、计算复杂性和性能之间找到正确的平衡是一项具有挑战性的任务，这需要仔细考虑和评估各种算法和系统级技术。
## 3 Taxonomy
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/Pasted-image-20241009150811.58hcqf64j1.webp)
### Algorithmic Innovation
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/Pasted-image-20241009151413.4jo36eilim.webp)
#### Decoding Algorithm
- **非自回归解码**
	idea: 放弃自回归生成范式，并行解码输出token。在解码过程中打破单词依赖性，并假设一定程度的条件独立性。
	work: 
		Parallel Decoding of Conditional Masked Language Models.
		Non-autoregressive neural machine translation
		Non-autoregressive neural machine translation with enhanced decoder input.
	optimization idea: 通过建模输出依赖性或迭代细化输出令牌，以达到自回归模型的质量。
	work: 
		Semi-autoregressive training improves mask-predict decoding.
		Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade
		Improving Non-autoregressive Translation with Dependency-Aware Decoder.
		Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement.
	optimization idea: 块状并行解码在基础LLM中插入一个单一的前馈层，以并行预测多个未来位置，然后回退到由基础模型验证的最长前缀。最近的一些努力致力于在一步解码中生成多个令牌，而无需对模型进行任何训练或修改。
	work: 
		Blockwise parallel decoding for deep autoregressive models.
		Accelerating Transformer Inference for Translation via Parallel Decoding.
		
	survey:
		A survey on non autoregressive generation for neural machine translation and beyond.

- 推测性解码
	idea: 推测性执行来应对顺序执行的限制，并提高解码的并行性。在自回归LLM推理过程中的每个解码步骤都可以被视为执行一个带有条件分支的程序。这个方法来自于这样一个事实：预测的输出总是由原始LLM验证，而当预测出错时，回退机制会生效。
	work:
		Speculative computation, parallelism, and functional programming.
		Accelerating large language model decoding with speculative sampling.
		Fast inference from transformers via speculative decoding.
	optimization idea: 通过引入多个小型草稿模型，并结合一种新颖的基于树的推测性推理和token验证机制。
	work: 
		SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.
	回退机制：
		Big little transformer decoder

- 早期退出机制
	idea: 基于早期模型层的输出有可能自信地推断出目标分布。它们可以基于内部分类器发出预测，而不是运行整个LLM。
	work: 
		Fast inference via early exiting from deep neural networks.
	退出条件:
		Magic pyramid: Accelerating inference with early exiting and token pruning.
		Accelerating Inference for Pretrained Language Models by Unified Multi-Perspective Early Exiting
		A global past-future early exit method for accelerating inference of pre-trained language models.
		FastBERT: a Self-distilling BERT with Adaptive Inference Time
		A simple hash-based early exiting approach for language understanding and generation
		DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference
		TR-BERT: Dynamic Token Reduction for Accelerating BERTInference
		Learning to Skip for Language Modeling.
		Bert loses patience: Fast and robust inference with early exit.
	自适应计算:
		SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference.
		Consistent Accelerated Inference via Confident Adaptive Transformers.

- 级联推理
	idea: 级联推理采用一系列规模不同的大型语言模型（LLMs）来最小化响应时间。将它们以级联方式组织起来，并根据实例难度自适应地选择合适的分类器。
	work: 
		Cascadebert: Accelerating inference of pre-trained language models via calibrated complete models cascade.
		Tabi: An Efficient Multi-Level Inference System for Large Language Models.
		FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.
		On Optimal Caching and Model Multiplexing for Large Model Inference.
		LARGE LANGUAGE MODEL CASCADES WITH MIXTURE OF THOUGHT REPRESENTATIONS FOR COST-EFFICIENT REASONING
		Chain-of-thought prompting elicits reasoning in large language models.
		Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.
#### Architecture Design
work: Simplifying Transformer Blocks.
- 配置缩减：
	idea: 缩减模型配置，使用浅层编码器或解码器
	work:
		PoWER-BERT: Accelerating BERT inference via progressive word-vector elimination.
		Adapler: Speeding up inference by adaptive length reduction.
	optimization idea: 权重共享，词汇表缩减
		Shallow Decoder: Reevaluating Non-autoregressive Machine Translation.





### System Optimization


## 4 Software Frameworks


## 5 Benchmarks



## 6 Connection with other surveys





## 7 Future Direction

