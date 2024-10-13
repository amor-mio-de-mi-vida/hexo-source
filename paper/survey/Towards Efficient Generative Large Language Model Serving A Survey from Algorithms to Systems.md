---
date: 2024-10-09 13:14:45
date modified: 2024-10-13 21:18:30
title: "Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems"
tags:
  - paper
  - survey
categories: " paper"
---
LLM 的空前成功带来了一些挑战，最明显的是它们在服务期间的巨大计算要求。巨大的模型大小和复杂性，加上对大量计算资源的需求，阻碍了它们在实际应用中的广泛部署。这些模型的资源密集型性质引发了对能耗、可扩展性和可访问性的担忧，阻碍了它们在没有丰富计算资源的更广泛社区中的采用。

本调查的主要目的是全面概述 LLM 服务和推理的最新进展。我们将根据现有技术的基本方法对其进行系统回顾和分类，突出它们的优势和局限性。该调查将涵盖广泛的方法，包括解码算法、架构设计、模型压缩、低位量化、并行计算、内存管理、请求调度和内核优化。

[Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems (arxiv.org)](https://arxiv.org/pdf/2312.15234)

<!-- more -->

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
- 
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
	
	`idea`: 通过引入多个小型草稿模型，并结合一种新颖的基于树的推测性推理和token验证机制。
	
	SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.
	
	回退机制：Big little transformer decoder

- 早期退出机制

	`idea`: 基于早期模型层的输出有可能自信地推断出目标分布。它们可以基于内部分类器发出预测，而不是运行整个LLM。
	
	Fast inference via early exiting from deep neural networks.
	
	Magic pyramid: Accelerating inference with early exiting and token pruning.
	
	Accelerating Inference for Pretrained Language Models by Unified Multi-Perspective Early Exiting.
	
	A global past-future early exit method for accelerating inference of pre-trained language models.
	
	FastBERT: a Self-distilling BERT with Adaptive Inference Time.
	
	A simple hash-based early exiting approach for language understanding and generation.
	
	DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.
	
	TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference.
	
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
	
	LARGE LANGUAGE MODEL CASCADES WITH MIXTURE OF THOUGHT REPRESENTATIONS FOR COST-EFFICIENT REASONING.
	
	Chain-of-thought prompting elicits reasoning in large language models.
	
	Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.

#### Architecture Design
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
	
	Transformers are rnns: Fast autoregressive transformers with linear attention.
	
	Linformer: Self-attention with linear complexity.
	
	`idea`: 借鉴先前的注意力简化方法，将它们概括和结合，以缩短上下文并减少KV缓存的大小，以及降低注意力复杂度
	
	Efficient Long-Range Transformers: You Need to Attend More, but Not Necessarily at Every Layer.
	
	Mistral 7B.
	
	Faster Causal Attention Over Large Sequences Through Sparse Flash Attention.
	
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
	
	Patient Knowledge Distillation for BERT Model Compression.
	
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

#### Low-bit Quantization：

`idea`:通过使用更少的比特（即少于32比特）来表示数值，一种方法是对LLM进行量化。

`survey`: A comprehensive study on post-training quantization for large language models.

`idea`: 量化感知训练与训练后量化，PTQ通过使用自定义的CUDA内核或编译将模型权重的计算精度甚至激活值降低到INT8或INT4。

A Speed Odyssey for Deployable Quantization of LLMs.

nuqmm: Quantized matmul for efficient inference of large-scale generative language models.

Atom: Low-bit Quantization for Efficient and Accurate LLM Serving.

LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.

SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.

Gptq: Accurate post-training quantization for generative pre-trained transformers.

OPTQ: Accurate quantization for generative pre-trained transformers.

AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.

Smoothquant: Accurate and efficient post-training quantization for large language models.

Zeroquant: Efficient and affordable post-training quantization for large-scale transformers.

RPTQ: Reorder-based Post-training Quantization for Large Language Models.

UnderstandingINT4Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases.

SqueezeLLM: Dense-and-Sparse Quantization.

Qlora: Efficient finetuning of quantized llms.

LLM-QAT: Data-Free Quantization Aware Training for Large Language Models.

The case for 4-bit precision: k-bit Inference Scaling Laws.

CacheGen: Fast Context Loading for Language Model Applications.

Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization.

#### Parallel Computation.

- 模型并行：
	
	`idea`: 张量模型并行将模型层（例如，注意力、FFN）从内部维度（例如，头、隐藏层）分割成多个部分，并在单独的设备（例如，GPU）上部署每个部分。
	
	Megatron-lm: Training multi-billion parameter language models using model parallelism.
	
	Efficiently scaling transformer inference.
	
	SUMMA: Scalable universal matrix multiplication algorithm.
	
	`idea`: 管道模型并行将模型层按顺序排列在多个设备上。每个设备负责一个管道阶段，该阶段由多个连续的模型层组成。
	
	Memory-efficient pipeline parallel dnn training.
	
	`idea`: 序列并行有各种差异化的设计和实现，但其核心思想是通过对长序列的处理在多个GPU之间进行分割，从而分布式计算和存储负载。
	
	Ring Attention with Blockwise Transformers for Near-Infinite Context.
	
	Calculon: a methodology and tool for high-level co-design of systems and large language models.
	
	`idea`: 自动并行用于分布式训练，通过替换它们的成本模型以适应Transformer模型的可预测运行时，可以轻松地将先前的自动搜索算法（例如，动态规划，整数线性规划）应用于LLM服务，并在无需手动干预的情况下确定最有效的并行策略。
	
	Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning.
	
	Beyond Data and Model Parallelism for Deep Neural Networks.
	
	Unity: Accelerating {DNN} training through joint optimization of algebraic transformations and parallelization.
	
	Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.
	
	Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models.
	
	AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.
	
	FlexFlow-Serve. https://github.com/Flexflow/FlexFlow/tree/inference. Commit: 672cdad, Accessed on: 2023-11 25.
	
	SpotServe: Serving Generative Large Language Models on Preemptible Instances.
	
	`idea`: 使能卸载技术，除了有限的设备内存（例如，GPU DRAM）之外，还使用更大但更慢的内存（例如，CPU DRAM）来保存模型参数和KV缓存。
	
	LLM in a flash: Efficient Large Language Model Inference with Limited Memory.
	
	Deep speed inference: Enabling efficient inference of transformer models at unprecedented scale.
	
	STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining.
	
	SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.
	
	FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.
	
- 去中心化推理：
	
	`idea`: 这种方法涉及模型和数据并行主义的结合，其中多个去中心化的自愿节点协作处理数据并推断输出。这种方法在硬件资源地理分布的场景中特别有用。
	
	Petals: Collaborative inference and fine-tuning of large models.
	
	HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment.
	
	Distributed Inference and Fine-tuning of Large Language Models Over The Internet.
	
	FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs.

#### Memory Management

高效的内存管理仍然是LLM服务中的首要挑战，特别是考虑到变压器架构固有的内存密集型特性。随着对长序列推理需求的增长，KV缓存的内存占用成为了相比于模型权重和其他激活所需工作空间的主要优化目标。

Efficient Memory Management for Large Language Model Serving with PagedAttention.

SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.

LightLLM. https://github.com/ModelTC/lightllm.

显而易见，LLM推理中的内存减少与其它算法创新和系统级优化紧密相关。虽然某些方法可能适用于特定的工作负载，但它们可能会相互抵消，导致整体性能下降。在LLM推理系统的内存效率和计算性能之间找到正确的平衡仍然是一个开放且紧迫的挑战。

#### Request Scheduling

有效地调度传入的推理请求对于优化LLM服务至关重要，这些算法旨在最大化资源利用率，保证在延迟服务水平目标（SLO）内的响应时间，并有效处理变化的需求负载。高效地管理传入请求并优化资源利用率。

Batch: machine learning inference serving on serverless platforms with adaptive batching.

Microsecond-scale preemption for concurrent GPU-accelerated DNN inferences.

Paella: Low-latency Model Serving with Software defined GPU Scheduling.

PipeSwitch: Fast pipelined context switching for deep learning applications.

Cocktail: A multidimensional optimization for model serving in cloud.

MArk: Exploiting Cloud Services for  Machine Learning Inference Serving.

考虑到可变的输出序列长度，它以首次到达优先（FCFS）的顺序在迭代级别调度引擎的执行，并允许对选定的操作集进行批处理以更好地利用硬件。

Orca: A Distributed Serving System for Transformer-Based Generative Models.

RayLLM. https://github.com/ray-project/ray-llm.

NVIDIA TensorRT-LLM. https://github.com/NVIDIA/TensorRT-LLM.

Fast Distributed Inference Serving for Large Language Models.

SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.

DeepSpeed-FastGen. https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen.

S3: Increasing GPU Utilization during Generative Inference for Higher Throughput.

#### Kernel Optimization

- 内核融合：
	
	`idea`: 为了减少内核启动和内存访问的开销，内核融合被之前的深度神经网络框架和编译器广泛采用。由于LLM推理不需要反向计算，因此存在更多的内核融合机会。
	
	NVIDIA Faster Transformer. https://github.com/NVIDIA/FasterTransformer.
	
	TenTrans High-Performance Inference Toolkit for WMT2021 Efficiency Task.
	
	Turbotransformers: an efficient gpu serving system for transformer models.
	
	LightSeq: A high performance inference library for transformers.
	
	A high-performance transformer boosted for variable-length inputs.
	
	Welder: Scheduling Deep Learning Memory Access via Tile-graph.
	
- 定制化注意力：
	
	`idea`: 为了使注意力操作在GPU上高效运行，专门为注意力计算定制GPU内核是至关重要的。
	
	NVIDIA cuDNN MultiHeadAttn. https://docs.nvidia.com/deeplearning/cudnn/api/index.html# cudnnMultiHeadAttnForward.
	
	`idea`: 用于第一次迭代（即初始/预填充/上下文/提示阶段），它并行处理输入提示中的所有token。
	
	xFormers: A modular and hackable Transformer modelling library. https://github.com/facebookresearch/xformers.
	
	NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass.
	
	Accelerating transformer networks through recomposing softmax layers.
	
	Online normalizer calculation for softmax.
	
	Self-attention Does Not Need $O(𝑛^2)$ Memory.
	
	用于后续迭代（即增量/解码/生成阶段），每个迭代只生成一个输出token的内核。
	
	`idea`: 对于自回归解码，常见的做法是保存先前计算过的键和值，这样在生成新令牌时只需要计算一个查询，而不是重新运行整个序列。这个领域优化的主要方向是最大化线程占用率并最小化设备上的高带宽内存。
	
	Et: re-thinking self-attention for transformer models on gpus.
	
	Flash-Decoding for long-context inference.
	
	FlashDecoding++: Faster Large Language Model Inference on GPUs.
	
	根据工作负载选择合适的并行维度对于更好的线程利用率是必要的。
	
- 采样优化：
	
	并行采样技术，如束搜索（beam search），通过在每次迭代中维护固定数量（即束宽）的最高分序列，有效地解码近似最优序列
	
	`idea`: 提出了多种随机采样技术来引入随机性，以获得更多样化的输出。
	
	Hierarchical Neural Story Generation.
	
	The curious case of neural text degeneration.
	
	Ctrl: A conditional transformer language model for controllable generation.
	
	`idea`: 由于冗余的KV缓存导致的内存压力增加，并且LLM的大词汇量（即数以万计）导致的采样效率问题。
	
	LightSeq: A high performance inference library for transformers.
	
- 可变序列长度：
	
	`idea`: LLM推理的另一个独特挑战是序列在输入长度和输出长度上可以变化，且后者是预先未知的。一种加快推理速度的方法是一次处理多个序列的批次。然而，当一批序列具有可变的输入长度时，通常会使用填充（padding）来使它们在批量处理时长度相同，这样做浪费了计算和内存资源。
	
	NVIDIA Effective Transformer. https://github.com/bytedance/effective_transformer.
	
	Bytetransformer: A high-performance transformer boosted for variable-length inputs.
	
	The CoRa tensor compiler: Compilation for ragged tensors with minimal padding.
	
	Improving Computation and Memory Efficiency for Real-world Transformer Inference on GPUs.
	
	SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.
	
- 自动编译
	
	大多数现有的LLM推理系统使用特定供应商的库作为其后端，例如cuBLAS、cuDNN和CUTLASS，这些库提供了优化的内核实现。为了进一步提高推理效率，它们还付出了巨大努力来为特定的LLM运算符（例如，注意力）在NVIDIA GPU上手动编写优化的内核。尽管有这些工作，使用自动化DNN编译器的趋势仍然存在。
	
	Apache TVM Unity: a vision for the ML software and hardware ecosystem.
	
	Relax: Composable Abstractions for End-to-End Dynamic Machine Learning.
	
	Tensorir: An abstraction for automatic tensorized program optimization.
	
	SparseTIR: Composable abstractions for sparse compilation in deep learning.
	
	MLIR-based code generation for GPU tensor cores.
	
	Compiling machine learning programs via high-level tracing.
	
	Triton: an intermediate language and compiler for tiled neural network computations.
	
	TASO: optimiz ing deep learning computation with automatic generation of graph substitutions.
	
	PyTorch 2.0: The Journey to Bringing Compiler Technologies to the Core of PyTorch.
	
	EINNET: Optimizing Tensor Programs with Derivation-Based Transformations.

## Software Frameworks

![Comparison of state-of-the-art open-sourced GPU-based LLM serving systems.](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.2krwjy3ire.webp)

## Benchmarks



## Connection with other surveys

我们的调研在高效生成式大型语言模型（LLM）服务和推理方面，补充并拓展了现有领域文献的范围，同时保持了独特的关注点。在相关工作中，[144]的研究主题与我们的调研最为接近，它探讨了更通用的Transformer模型和特定领域的加速器设计。然而，我们的调研通过专门针对生成式LLM服务这一精细领域，与其他研究区分开来，这一领域尚未成为其他研究的中心。此外，一些研究深入进行了LLM在GPU上推理效率[190, 297]和新型加速器[78]的实验性研究，提供了直接关联我们服务效率研究的宝贵实证见解。此外，LLMCarbon [79] 关注了LLM部署中越来越重要的一个方面——对环境的影响（例如，碳足迹）。尽管我们的调研主要从性能角度关注效率，但这类研究提供的环境视角无疑在我们的广泛讨论中是相关且值得尊敬的。一些调研和基准测试[126]提供了关于模型压缩[113, 248, 314, 314]和量化[99, 280]的宝贵见解，这些研究为我们的相关方向探索间接提供了基础。一些研究[65, 187]为理解LLM的有效性（例如，准确性、困惑度、事实性等）提供了必要的背景，这超出了我们调研的范围。我们的调研也认可了先前专注于大规模深度神经网络（DNN）模型分布式训练的调研[42, 175]的贡献，因为它们为考虑LLM服务提供了背景信息。从本质上讲，我们的调研位于众多研究之中，从中吸取并贡献了对LLM服务效率更全面的理解，包括算法创新和系统优化。通过整合这些领域的见解，我们旨在提供一个细致而全面的概述，涵盖该领域最新的进展和挑战。

Full stack optimization of transformer inference: a survey.

Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models.

Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models.

A Comprehensive Performance Study of Large Language Models on Novel AI Accelerators.

LLMCarbon: Modeling the end-to-end Carbon Footprint of Large Language Models.

Compressing LLMs: The Truth is Rarely Pure and Never Simple.

Compression of deep learning models for text: A survey.

Efficient methods for natural language processing: A survey.

A survey on model compression for large language models.

A survey of quantization methods for efficient neural network inference.

A comprehensive study on post-training quantization for large language models.

LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking.

Generating benchmarks for factuality evaluation of language models.

Demystifying parallel and distributed deep learning: An in-depth concurrency analysis.

Scalable deep learning on distributed infrastructures: Challenges, techniques, and tools.

## Future Direction

### 硬件加速器的发展与增强

提高生成式大型语言模型（LLM）服务效率的未来进展，可能会在很大程度上依赖于专门硬件加速器的开发和完善，以及与硬件和软件优化相协调的共同设计方法。

例如，将内存更紧密地集成到处理单元附近，或优化芯片架构以更好地适应LLM算法的数据流，可以显著降低延迟和能耗。这一方法在最近的GPU发展中已有体现，如NVIDIA的Hopper架构，它在HBM和SRAM容量、内存带宽、计算单元和分割带宽方面取得了改进，直接有利于LLM的处理。

在这一领域的持续创新可能包括设计本质上针对生成式LLM计算模式的硬件，比如针对这些模型中常见的注意力机制和张量操作的具体需求进行优化，最终影响LLM服务系统的设计和实施。

NVIDIA H100 Tensor Core GPU Architecture. https://resources.nvidia.com/en-us-tensor-core/gtc22 whitepaper-hopper.

### 高效且有效的解码算法

开发更高效的解码算法可以大幅提升服务的效率。鉴于对更有效地利用LLM中所包含的丰富知识的迫切需求，未来的研究可以探索不同于传统自回归方法的新途径，以实现实时应用中的生成速度提升，同时保持解码质量。

一个充满希望的研究方向是广义推测推理，因为它能够在保持生成质量的同时提高效率。具体而言，可以将小型推测模型泛化到任何能够比LLM更高效地生成初步令牌的其他方法，例如知识检索器和用户定义的函数。例如，最近的一些研究工作开始使用早期退出或非自回归解码来替代初步模型。

总结来说，开发像推测解码这样的高效解码算法，并结合底层系统的优化，是提升生成式LLM服务效率的一个重要机遇。

SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.

Inference with reference: Lossless acceleration of large language models.

Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding.

SPEED: Speculative Pipelined Execution for Efficient Decoding.

Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding.

Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding.

Breaking the Sequential Dependency of LLM Inference Using Lookahead Decoding.

Lossless acceleration for Seq2seq generation with aggressive decoding.

### 长上下文/序列场景的优化。

随着LLM的应用向更复杂的场景扩展，处理更长上下文或序列的需求持续上升。在处理长序列工作负载的LLM服务中，需要从算法和系统两个方面解决挑战。对于LLM来说，当序列长度超过训练期间所观察到的长度时，它们常常会出现长度泛化失效的问题，即使启用了相对位置编码或在更长的语料库上进行微调之后。即便是某些声称支持超长上下文的模型，研究也发现它们会遇到“中间损失”的问题。当前的方法试图通过减少计算序列长度的同时保留相关信息来减轻这些限制，比如采用检索增强、序列压缩和缓存等技术。对于LLM服务系统来说，长序列带来了重要挑战，包括增加的内存消耗、KV缓存的访问以及自注意力计算复杂性的成倍增加。

Test Long: Attention with Linear Biases Enables Input Length Extrapolation.

Extending context window of large language models via positional interpolation.

LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.

Lost in the middle: How language models use long contexts.

Retrieval meets Long Context Large Language Models.

LongLLM Lingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.

Prompt Cache: Modular Attention Reuse for Low-Latency Inference.

### 探究替代架构

尽管目前Transformer模型和自注意力机制在大型语言模型（LLM）领域占据主导地位，但是探索替代架构是未来研究的一个充满希望的方向。在深度学习（DL）领域的历史上，我们见证了主导架构的持续更迭，每一次范式的转变都会带来重大的进步。基于这种趋势，考虑其他可能带来独特优势的架构方法尤为重要，特别是在提高计算效率方面。例如，一些最新的研究正在探索无注意力方法，使用纯多层感知器（MLP）架构来替代注意力机制。深度神经网络模型架构的演变不仅是一种自然的发展过程，也是为了发现更高效和有效的LLM结构方式的必要探索。

Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers.

### 探索在复杂环境中的部署

随着大型语言模型（LLM）应用的不断扩展，一个至关重要的未来方向是探索和优化它们在不同复杂环境中的部署。这种探索不仅限于传统的基于云的部署，还包括边缘计算、混合计算（结合云和边缘计算）、去中心化计算，以及利用更经济的资源，如抢占式实例。每一种环境都为LLM服务带来了独特的挑战和机遇。例如，边缘计算通过在数据源附近处理数据，可以实现更快的响应时间和减少带宽消耗，但同时它也面临着计算资源和存储容量有限的问题。混合计算提供了一种平衡的方法，但需要先进的管理策略来有效地分配计算任务。去中心化计算为众包计算资源提供了一条有希望的道路，但也带来了数据隐私和安全的额外考量。在抢占式资源上提供LLM服务可以显著降低成本，但需要容错机制来应对其固有的不可预测性和变异性，以确保性能的稳定和系统的可靠性。成功应对这些复杂环境中的挑战，将是实现更加强大、可扩展且高效的LLM应用的关键。

The future of AI is hybrid. https://www.qualcomm.com/content/dam/qcomm-martech/dm assets/documents/Whitepaper-The-future-of-AI-is-hybrid-Part-2-Qualcomm-is-uniquely-positioned-to-scale hybrid-AI.pdf.

BumbleBee: Secure Two-party Inference Framework for Large Transformers.

LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud.

SpotServe: Serving Generative Large Language Models on Preemptible Instances.

### 自动适应特定需求

多样化的应用特定需求创造了广泛的创新LLM服务优化机遇，例如参数高效微调，从外部向量存储中进行检索，在线学习和知识更新，多模态工作负载，以及将不同LLM的能力串联起来。这些独特的挑战也要求能够自动且无缝地将LLM服务技术集成到现有的IT基础设施中，通过将优化范围扩展到整个LLM生命周期，包括数据采集和处理，自动机器学习（AutoML）和模型管理，资源分配，以及性能监控。

Punica: Multi-Tenant LoRA Serving.

S-LoRA: Serving Thousands of Concurrent LoRA Adapters.

PetS: A Unified Framework for Parameter Efficient Transformers Serving.

Improving language models by retrieving from trillions of tokens.

Autogen: Enabling next-gen llm applications via multi-agent conversation framework.

AutoML in the Age of Large Language Models: Current Challenges, Future Opportunities and Risks.

Saturn: An Optimized Data System for Large Model Deep Learning Workloads.

