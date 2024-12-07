---
date: 2024-12-02 19:28:32
date modified: 2024-12-03 11:24:28
title: ZeRO Memory Optimizations Toward Training Trillion Parameter Models
tags: 
categories: []
date created: 2024-09-25 13:26:08
---

**Abstract**: 大型深度学习模型提供了显著的准确性提升，但训练数十亿到数万亿参数是具有挑战性的。现有的解决方案如数据和模型并行化在将模型适应有限的设备内存方面存在基本限制，同时获得计算、通信和开发效率。我们开发了一种新颖的解决方案，<mark>零冗余优化器（ZeRO），以优化内存，极大地提高了训练速度，同时增加了可以高效训练的模型大小。ZeRO消除了数据和模型并行训练中的内存冗余，同时保持了低通信量和高的计算粒度，使我们能够根据设备数量按比例扩展模型大小，并保持持续的高效率。</mark>我们对内存需求和通信量的分析表明：ZeRO有潜力使用今天的硬件扩展到超过1万亿参数。 

我们实现了并评估了ZeRO：它在400个GPU上以超线性速度训练超过1000亿参数的大型模型，实现了15 Petaflops的吞吐量。这代表了模型大小增加了8倍，相比最先进的实现，性能提高了10倍。在可用性方面，ZeRO可以在不要求模型并行化的情况下训练大型模型，最多可达130亿参数（例如，大于Megatron GPT 8.3B和T5 11B），这对于科学家来说更难应用。最后但同样重要的是，研究人员已经使用ZeRO的系统突破创建了世界上最大的语言模型（170亿参数），具有创纪录的准确性。

## Key findings

Data Parallelism, Pipeline Parallelism, Model Parallelism, CPU-Offloading are common technique for model training.

Among different existing solution for training large models, MP is perhaps the most promising, but it will brings significant communication between each layer and is limited to the model. (which can not scale further)

For large models, the majority of the memory is occupied by *model states* which include the optimizer states (such as momentum and variances in Adam), gradients, and parameters. The remaining memory is consumed by activation, temporary buffers and unusable fragmented memory, which we refer to collectively as *residual states.*

### Optimizing Model State Memory

#### What we observed

Data Parallelism replicates the entire model states across all data parallel process resulting in redundant memory consumption;

Model Parallelism partition these states to obtain high memory efficiency, but often result in too fine-grained computation and expensive communication that is less scaling efficient.

All of these approaches maintain all the model states required over the entire training process statically, even though not all model states are required all the time during the training.


#### ZeRO-DP

- Optimizer State Partitioning ($P_{os}$): 4x memory reduction, same communication volume as DP.

- Add Gradient Partitioning ($P_{os+g}$): 8x memory reduction, same communication volume as DP.

- Add Parameter Partitioning ($P_{os+g+p}$): Memory reduction is linear with DP degree $N_d$. For example, splitting across 64 GPUs will yield a 64x memory reduction. There is a modest 50% increase in communication volume.

![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/paper/image.7ax7i0w2bg.webp)

### Optimizing Residual State Memory

#### What we observed

- For activations (stored from forward pass inn order to perform backward pass), **checkpointing helps but not sufficient for large models.**

- fragmented memory during training **due to variations in the lifetime of different tensors**. Lack of contiguous memory due to fragmentation can cause memory allocation failure, even when enough free memory is available.

#### ZeRO-R

-   ZeRO-R optimizes by **identifying and removing activation replication in existing MP approaches** through activation partitioning. It also offloads activations to CPU when appropriate.

- ZeRO-R **defines appropriate size for temporary buffers** to strike for a balance of memory and computation efficiency.

- ZeRO-R proactively manages memory based on the different lifetime of tensors, preventing memory fragmentation.

#### Cases we want to leverage Model Parallelism

- When used with ZeRO-R, MP can reduce activation memory footprint for very large models.

- For smaller models where activation memory is not an issue, MP can also have benefits **when aggregated batch size using Data Parallelism alone is too big to have good convergence.**

## Related Work

#### Data, Model and Pipeline Parallelism

In DP, model parameters are replicated on each device. At each step, a mini-batch is divided evenly across all the data parallel processes, such that each process executes the forward and backward propagation on a different subset of data samples, and uses averaged gradients across processes to update the model locally.

<mark>Model Parallelism</mark>

<mark>Pipeline Parallelism</mark> splits a model horizontally across layers running each partition on a different device and use micro-batching to hide the pipeline <mark>bubble</mark>. (tied-weights and batch-normalization are difficult to implement due to horizontal splitting and micro-batching)

**Solution**: G-pipe partitions both model parameters and total activations but requires a batch size proportional to number of pipeline partitions to hid the pipeline bubble. **Challenges**: The large batch size can affect the convergence rate, while also requiring significant memory to store activations.

**Solution**: PipeDream keeps multiple copies of stale parameters to hide the pipeline bubble without increasing the batch size significantly. **Challenges**: making it less memory efficient. Additionally the implementation is not equivalent to the standard DL training and has implications on training convergence. 

#### Non-parallelism based approach to reduce memory

- **Reducing Activation Memory**: compression, activation checkpointing, live analysis. 

- **CPU Offload**: offloading model states to CPU memory through algorithmic design or virtualized memory. (**Challenges**: up to 50% of training time can be spent on GPU-CPU-GPU transfers.) ZeRO-R may offload just the activation checkpoints for very large models to improve performance.

- **Memory Efficient Optimizer**: reducing memory consumption of adaptive optimization methods by maintaining coarser-grained statistics of model parameters and gradients, with potential impact on model convergence guarantees. 

### Training Optimizers
  
<mark>Adaptive optimization methods</mark> are crucial to achieving SOTA performance and accuracy for effective model training of large models. (**Challenges**: Compared to SGD, by maintaining fine-grained first-order and second-order statistics for each model parameter and gradient at the cost of significant memory footprint.)

#### Reference

### Where Did All the Memory Go?

#### Model States: Optimizer States, Gradients and Parameters

- **Mixed-Precision Training**: The state-of-the-art approach to train large models on the current generation of NVIDIA GPUs is via <mark>mixed precision (fp16/32) training</mark>, where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units on these GPUs. During mixed-precision training, both the forward and backward propagation are performed using fp16 weights and activations. However, to effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.

#### Residual Memory Consumption

- **Activations** can take up a significant amount of memory during training. Activation checkpointing (or activation recomputation) is a common approach to reduce the activation memory by approximately the square root of the total activations. (**Challenges**: it can still grow quite large for bigger models.

- **Temporary buffers** used for storing intermediate results consumes non-trivial amount of memory for large models. Operations such as gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer before applying the operation in an effort to improve throughput. (**Challenges**: While the gradient themselves are usually stored as fp16 tensors, the fused buffer can be an fp32 tensor depending on the operation. When the size of the model is large, these temporary buffer sizes are non-trivial.)

- **Memory Fragmentation**: A request for a memory will fail if there isn't enough contiguous memory to satisfy it, even if the total available memory is larger than requested. 


## Details

### Deep dive into ZeRO-DP

#### $P_{os}$: Optimizer State Partitioning

For a DP degree of $N_d$, we group the optimizer states into $N_d$ equal partitions, such that the

$\text{i}^\text{th}$ data parallel process only updates the optimizer states corresponding to the $\text{i}^\text{th}$ partition. Thus, each data parallel process only needs to store and update $1/N_d$ of the total optimizer states and then only update $1/N_d$ of the parameters. <mark>We perform an all-gather across the data parallel process at the end of each training step to get the fully updated parameters across all data parallel process.</mark>
#### $P_g$: Gradient Partitioning

As each data parallel process only updates its corresponding parameter partition, it only needs the reduced gradients for the corresponding parameters. Therefore, as each gradient of each layer becomes available during the backward propagation, we only reduce them on the data parallel process responsible for updating the corresponding parameters. After the reduction we no longer need the gradients and their memory can be released. This reduces the memory footprint required to hold the gradients from $2Ψ$ bytes to $2Ψ/N_d$.

Effectively this is a Reduce-Scatter operation, where gradients corresponding to different parameters are reduced to different process. To make this more efficient in practice, we use a bucketization strategy, where we bucketize all the gradients corresponding to a particular partition, and perform reduction on the entire bucket at once. This is similar in spirit to how NVIDIA’s AMP optimizer bucketizes the all-reduce gradient computation to overlap communication and computation. In our case we perform a reduction instead of an all-reduce at the partition boundaries to reduce memory footprint and overlap computation and communication.

#### $P_p$: Parameter Partitioning

Just as with the optimizer states, and the gradients, each process only stores the parameters corresponding to its partition. When the parameters outside of its partition are required for forward and backward propagation, they are received from the appropriate data parallel process through broadcast. While this may seem to incur significant communication overhead at first glance, we show that this approach only increases the total communication volume of a baseline DP system to 1.5x, while enabling memory reduction proportional to $N_d$.

### Deep Dive into ZeRO-R

#### $P_a$: Partitioned Activation Checkpointing

As discussed in 4.2, MP by design requires a replication of the activations, resulting in redundant copies of the activations across model parallel GPUs. **ZeRO eliminates this redundancy by partitioning the activations, and only materializes them in a replicated form one activation layer at a time, right before the activation is used in computation.** More specifically, once the forward propagation for a layer of a model is computed, the input activations are partitioned across all the model parallel process, until it is needed again during the backprogation. At this point, ZeRO uses an all-gather operation to re-materialize a replicated copy of the activations. We refer to this optimization as Pa. It works in conjunction with activation checkpointing

[7], storing partitioned activation checkpoints only instead of replicated copies. Furthermore,

in the case of very large models and very limited device memory, these partitioned activation

checkpoints can also be offloaded to the CPU reducing the activation memory overhead to

nearly zero at an additional communication cost, which we will discuss in 7. We refer to this

as Pa+cpu



#### $C_B$: Constant Size Buffers






#### $M_D$: Memory Defragmentation








Related work

- [1] [Samyam R, Jeff R, Olatunji R, Yuxiong H. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arxiv.org/pdf/1910.02054. 2019.](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.02054.pdf)  
    
- [2] [Turing-NLG: A 17-billion-parameter language model by Microsoft](https://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)  
    
- [3] [Rangan M, Junhua W. ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters. 2020.](https://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)  
    
- [4] [KDD 2020: Hands on Tutorials: Deep Speed -System optimizations enable training deep learning models](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DczgA-MbAdvA%26t%3D2550s)  
    
- [5] [Mohammad S, Mostofa P, Raul P, et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism arxiv.org/abs/1909.08053 .2019](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.08053)  
    
- [6] [Rangan M, Andrey P. ZeRO-Infinity and DeepSpeed: Unlocking unprecedented model scale for deep learning training. 2021](https://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)  
    
- [7] [Xu Q, Li S, Gong C, et al. An Efficient 2D Method for Training Super-Large Deep Learning Models[J]. arXiv preprint arXiv:2104.05343, 2021.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.05343)