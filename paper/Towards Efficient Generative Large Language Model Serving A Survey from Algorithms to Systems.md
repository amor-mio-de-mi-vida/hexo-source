---
date: 2024-10-09 13:14:45
date modified: 2024-10-10 23:10:11
title: "Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems"
tags: 
categories: 
excerpt: LLM 的空前成功也带来了一些挑战，最明显的是它们在服务期间的巨大计算要求。巨大的模型大小和复杂性，加上对大量计算资源的需求，阻碍了它们在实际应用中的广泛部署。这些模型的资源密集型性质引发了对能耗、可扩展性和可访问性的担忧，阻碍了它们在没有丰富计算资源的更广泛社区中的采用。
---
[Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems (arxiv.org)](https://arxiv.org/pdf/2312.15234)

## Abstract
LLM 的空前成功也带来了一些挑战，最明显的是它们在服务期间的巨大计算要求。巨大的模型大小和复杂性，加上对大量计算资源的需求，阻碍了它们在实际应用中的广泛部署。这些模型的资源密集型性质引发了对能耗、可扩展性和可访问性的担忧，阻碍了它们在没有丰富计算资源的更广泛社区中的采用。

本调查的主要目的是全面概述 LLM 服务和推理的最新进展。我们将根据现有技术的基本方法对其进行系统回顾和分类，突出它们的优势和局限性。该调查将涵盖广泛的方法，包括解码算法、架构设计、模型压缩、低位量化、并行计算、内存管理、请求调度和内核优化。
### Structure

## Background

### Transformer-based LLM

从数学上讲，Transformer 中的自注意力机制可以描述如下：对于输入序列
$$X=[x_1, x_2, ..., x_n]$$
Transformer 使用 $X$ 的线性变换计算一组查询 $Q$、键 $K$ 和 $valuesV$。然后，自我注意分数计算为：$$\text{Attention}(Q, K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$
其中$d_k$是键的维度。这种机制允许模型专注于输出每个元素的 input 序列的不同部分，捕获复杂的依赖关系，而不管它们在 input sequence 中的距离如何。

Transformer 中的另一个重要结构是前馈网络 （FFN），它存在于 Transformer 的每一层中，并显着影响其计算强度。FFN 通常由两个线性变换组成，中间有一个非线性激活函数，通常表示为：
$$\text{FFN}(x)=max(0, xW_1+b_1)W_2+b_2$$
其中，$W_1$、$W_2$、$b_1$ 和 $b_2$ 是 FFN 的可学习参数，非线性函数 $max（0，·）$（在本例中为 ReLU）将必要的非线性引入模型，使其能够学习更复杂的模式。FFN 负责模型参数计数的很大一部分，因此负责其内存占用和计算负载。在每个 Transformer 层中，在多头注意力 （MHA） 聚合来自输入不同部分的信息后，FFN 会为每个位置独立处理这些聚合信息。这种并行处理能力是 Transformer 的一个关键优势，使其能够有效地处理序列。但是，这也意味着计算负载和内存需求随输入序列的长度和网络的深度而变化。

在基于 Transformer 的 LLM 中，自我注意和 FFN 的结合使这些模型能够捕获广泛的语言上下文和细微差别，从而在各种 NLP 任务中设定新的基准。然而，训练和推理的大量计算要求已成为一个关键的研究领域，专注于在不显著影响性能的情况下优化这些方面。Transformer 模型还包括其他关键组件，如位置编码，它添加了有关序列中每个标记位置的信息，以及多头注意力机制，它允许模型关注不同表示空间中序列的不同部分。

### GPUs and Other Accelerators

### LLM Inference

先前的研究对基于Transformer的LLM推理的算法强度进行了深入分析（例如，计算浮点运算次数、I/O和内存消耗），并根据自回归解码算法的执行提供了广泛的实证结果进行成本估算（例如，建模推理延迟）。大型语言模型推理的优化是一个复杂的问题，因为可能存在不同的最优策略，不同的算法配置和系统设置。

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
	
## Taxonomy
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/Pasted-image-20241009150811.58hcqf64j1.webp)

### Algorithmic Innovation
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/Pasted-image-20241009151413.4jo36eilim.webp)

#### Decoding Algorithm
- **非自回归解码**
	`idea`: 放弃自回归生成范式，并行解码输出token。在解码过程中打破单词依赖性，并假设一定程度的条件独立性。
	Parallel Decoding of Conditional Masked Language Models.
	Non-autoregressive neural machine translation
	Non-autoregressive neural machine translation with enhanced decoder input.
	
	`idea`: 通过建模输出依赖性或迭代细化输出令牌，以达到自回归模型的质量。
	Semi-autoregressive training improves mask-predict decoding.
	Fully Non-autoregressive Neural Machine Translation: Tricks of the Trade
	Improving Non-autoregressive Translation with Dependency-Aware Decoder.
	Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement.
	
	`idea`: 块状并行解码在基础LLM中插入一个单一的前馈层，以并行预测多个未来位置，然后回退到由基础模型验证的最长前缀。最近的一些努力致力于在一步解码中生成多个令牌，而无需对模型进行任何训练或修改。
	Blockwise parallel decoding for deep autoregressive models.
	Accelerating Transformer Inference for Translation via Parallel Decoding.
		
	`survey`: A survey on non autoregressive generation for neural machine translation and beyond.

- 推测性解码
	`idea`: 推测性执行来应对顺序执行的限制，并提高解码的并行性。在自回归LLM推理过程中的每个解码步骤都可以被视为执行一个带有条件分支的程序。这个方法来自于这样一个事实：预测的输出总是由原始LLM验证，而当预测出错时，回退机制会生效。
	Speculative computation, parallelism, and functional programming.
	Accelerating large language model decoding with speculative sampling.
	Fast inference from transformers via speculative decoding.
	optimization idea: 通过引入多个小型草稿模型，并结合一种新颖的基于树的推测性推理和token验证机制。
	SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.
	回退机制：Big little transformer decoder

- 早期退出机制
	`idea`: 基于早期模型层的输出有可能自信地推断出目标分布。它们可以基于内部分类器发出预测，而不是运行整个LLM。
	Fast inference via early exiting from deep neural networks.
	Magic pyramid: Accelerating inference with early exiting and token pruning.
	Accelerating Inference for Pretrained Language Models by Unified Multi-Perspective Early Exiting
	A global past-future early exit method for accelerating inference of pre-trained language models.
	FastBERT: a Self-distilling BERT with Adaptive Inference Time
	A simple hash-based early exiting approach for language understanding and generation
	DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference
	TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference
	Learning to Skip for Language Modeling.
	Bert loses patience: Fast and robust inference with early exit.
	SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference.
	Consistent Accelerated Inference via Confident Adaptive Transformers.

- 级联推理
	`idea`: 级联推理采用一系列规模不同的大型语言模型（LLMs）来最小化响应时间。将它们以级联方式组织起来，并根据实例难度自适应地选择合适的分类器。
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
	`idea`: 缩减模型配置，使用浅层编码器或解码器
	PoWER-BERT: Accelerating BERT inference via progressive word-vector elimination.
	Adapler: Speeding up inference by adaptive length reduction.
	
	`idea`: 权重共享，词汇表缩减
	Shallow Decoder: Reevaluating Non-autoregressive Machine Translation.
- 注意力简化：
	自注意力计算的一个突出挑战是计算复杂度$O(𝐿^2)$，它与输入序列长度$𝐿$成二次方关系。为了应对非常长的序列任务，需要将这些标准注意力简化为更高效的选择。
	`survey` : Efficient Transformers: A Survey.
	Big bird: Transformers for longer sequences.
	Transformers are rnns: Fast autoregressive transformers with linear attention
	Linformer: Self-attention with linear complexity
	
	`idea`: 借鉴先前的注意力简化方法，将它们概括和结合，以缩短上下文并减少KV缓存的大小，以及降低注意力复杂度
	Efficient Long-Range Transformers: You Need to Attend More, but Not Necessarily at Every Layer.
	Mistral 7B.
	Faster Causal Attention Over Large Sequences Through Sparse Flash Attention
	Longnet: Scaling transformers to 1,000,000,000 tokens.
	
	`idea`: 通过将上下文压缩成更少的软token来进行上下文压缩
	Adapting Language Models to Compress Contexts.
	Landmark Attention: Random-Access Infinite Context Length for Transformers.
	In-context autoencoder for context compression in a large language model.
	CacheGen: Fast Context Loading for Language Model Applications.
	
	`idea`: 根据不同的重要性指导直接丢弃或重新表述不重要的上下文token
	Extending Context Window of Large Language Models via Semantic Compression.
	Llmlingua: Compressing prompts for accelerated inference of large language models.
	Compressing Context to Enhance Inference Efficiency of Large Language Models.
	Learning to compress prompts with gist tokens.
	Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers.
	Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.
	H\_2 O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.
	Efficient Streaming Language Models with Attention Sinks.
	Longformer: The long-document transformer.

表1展示了四种代表性方法的稀疏注意力模式及其应用。然而，由于上下文不完整，这些方法在实际工作负载中可能面临不可避免的信息损失，特别是当注意力分布更复杂时。
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.5q7eh98kv4.webp)
- 激活共享：通过共享中间激活来提高注意力计算的效率。
	`idea`: 注意力共享方法观察到不同层的注意力矩阵分布之间的相似性，并重新使用这些注意力矩阵以减少计算成本。
	An efficient transformer decoder with compressed sub-layers.
	Speeding up Transformer Decoding via an Attention Refinement Network.
	Sharing Attention Weights for Fast Transformer.
	
	`idea`: 多查询注意力(MQA)使得不同的头共享同一组键和值，以减少增量推理中的内存带宽需求。
	Fast transformer decoding: One write-head is all you need.
	
	`idea`: 组查询注意力(GQA)放宽了单一组键和值的限制到多组，并且每组与一组查询耦合。
	GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.
- 条件计算：
	稀疏激活的专家混合模型（MoE）范式将模型的能力分布在各种“专家”上，这些“专家”是更小的神经网络，每个专家专注于数据的不同子集。它允许系统仅根据某些路由机制调用给定输入所需的专家，而不是在整个大型模型上计算，从而实现了计算和内存效率。
	SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention.
	Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
	Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
	GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.
	Evomoe: An evolutional mixture-of-experts training framework via dense-to-sparse gate.
	Hash layers for large sparse models.
	Memory Augmented Language Models through Mixture of Word Experts.
	Mixture-of-experts with expert choice routing.
	Glam: Efficient scaling of language models with mixture-of-experts.
	
	Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference.
	MoE的动态特性也要求特殊的系统优化，包括分布式通信GPU内核实现，以促进MoE推理效率。
	FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models.
	Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.
	Tutel: Adaptive mixture-of-experts at scale.
	Accelerating Distributed {MoE} Training and Inference with Lina.
	FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement.
	Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale.
	MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.
	PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation.
- 循环单元
	尽管递归神经网络（RNN）在捕捉序列中的长期依赖关系方面存在困难，但仍有几种方法使用递归单元来替换Transformer模块，并在推理期间实现线性的计算和内存复杂度。
	RWKV: Reinventing RNNs for the Transformer Era.
	Retentive Network: A Successor to Transformer for Large Language Models.
	`idea`: 这些最近的探索大多建立在线性注意力表示之上。经过重组后，它们通过使用线性递归单元对token之间的交互进行建模，从而克服了注意力的$O(L^2)$瓶颈，这些递归单元更容易保持可并行化训练的性质。
	Transformers are rnns: Fast autoregressive transformers with linear attention.
	An attention free transformer.
	Hungry Hungry Hippos: Towards Language Modeling with State Space Models.
	Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
	Efficiently Modeling Long Sequences with Structured State Spaces.
	Long Range Language Modeling via Gated State Spaces.
	Resurrecting recurrent neural networks for long sequences.
	`idea`: 设计还包括各种位置编码模块，指数衰减机制以及一系列token级别的非线性MLPs或GLUs，以改进模型的表示能力。
	Roformer: Enhanced transformer with rotary position embedding.
	The statistical recurrent unit.
	Mlp-mixer: An all-mlp architecture for vision.
	Metaformer is actually what you need for vision.
	Language modeling with gated convolutional networks.
#### Model Compression
- 知识蒸馏：
	`idea`: 通过大型教师模型的监督来训练一个小型学生模型。大多数先前方法都在探索白盒蒸馏，这需要访问整个教师模型的参数。
	Knowledge Distillation of Large Language Models.
	TinyBERT: Distilling BERT for Natural Language Understanding.
	DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
	Patient Knowledge Distillation for BERT Model Compression
	Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers.
	`idea`: 黑盒蒸馏
	Stanford alpaca: An instruction-following llama model.
	Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality.
	Wizardlm: Empowering large language models to follow complex instructions.
	Instruction tuning with gpt-4.
	Minigpt-4: Enhancing vision-language understanding with advanced large language models.
	OpenAI. 2023. GPT-4 Technical Report.
- 网络剪枝: 
	结构化剪枝方法，这移除了整个结构化的LLM组件，便于GPU加速。
	Reducing Transformer Depth on Demand with Structured Dropout.
	Ziplm: Hardware-aware structured pruning of language models.
	LLM-Pruner: On the Structural Pruning of Large Language Models.
	What Matters In The Structured Pruning of Generative Language Models?
	Deja vu: Contextual sparsity for efficient llms at inference time.
	非结构化方法：它们通常实现50-60%的稀疏度以压缩LLMs。可以进一步泛化到半结构化的N:M稀疏（即2:4和4:8），利用NVIDIA稀疏张量核心的加速，显著提升推理速度。
	Accelerating sparse deep neural networks.
	LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation.
	DSFormer: Effective Compression of Text-Transformers by Dense-Sparse Weight Factorization.
	Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity.
	PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.

### System Optimization


## Software Frameworks


## Benchmarks



## Connection with other surveys





## Future Direction



