![image](https://github.com/user-attachments/assets/fa5c53c8-99fc-4720-a874-6f9210efd9bb)
<h2 align="center"> <a href="">大模型分布式加速训练方法基础知识总结</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
  
[![zhihu](https://img.shields.io/badge/知乎-0084FF)](https://blog.csdn.net/zwqjoy/article/details/130732601)
[![CSDN](https://img.shields.io/badge/CSDE-yellow)](https://zhuanlan.zhihu.com/p/660567767)

</h5>


# 1. 分布式通信术语
* **1. Broadcast：** 广播，一对多
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img_1.png)  
* **2. Reduce：** 各设备上相同位置的元素进行加和，并将结果呈现在一个设备上
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-2.png)  
* **3. All Reduce：** 相当于Reduce之后再来了一个Broadcast
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-3.png)  
* **4. Gather：** Gather的中文叫做收集（即把东西放到一起，并不做运算），与Reduce不同的地方是，Gather只是将数据汇总到一起，而Reduce需要“按照指定的映射函数进行运算”
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-4.png)   
* **5. All Gather：** 多对多广播 
* **6. Scatter：** 离散，扩散；即将一个机器上的不同数据分别给到不同机器。而广播的含义是将一个机器上的数据全部传输给其他机器
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-5.png) 
* **7. Reduce Scatter：** 先广播在加和；Reduce_scatter最终呈现效果为：每个GPU上有一块完整加和后的数据。他和All reduce的区别在于，All reduce是所有完整加和的数据。
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-6.png)  


参考文献：[https://blog.csdn.net/cy413026/article/details/138618053](https://blog.csdn.net/cy413026/article/details/138618053)

# 2. 分布式通信架构

* **2.1.All-reduce**  
     All-reduce架构中仅用到 GPU 机器，这是因为其设计假定了每个节点都是同构节点。迭代过程中，GPU 独立计算模型参数的梯度，然后使用 All-reduce 通信聚合梯度。  
* **2.2 PS 架构**   
     PS 则包含 GPU worker 和 CPU server。迭代过程中，GPU worker 将梯度传输至 CPU server；后者将接收到的不同 workers 的梯度做聚合，然后执行 DNN 优化器（如 RMSProp 或 Adam 等）并将更新后的参数传输回 GPU workers。
  ![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-7.png)     


# 3. 分布式训练框架
## 3.1 Pytorch原生支持DDP, FSDP

* **DDP**: 传统的数据并行, 每一个GPU卡上保存整个model的参数/梯度/优化器状态, 然后对数据集切分为 N个shard分片给不同的GPU进行训练，计算完梯度后通过all-reduce通信来做梯度的融合。

* **FSDP**: 全切片数据并行(Fully Sharded Data Parallel，简称为FSDP)是数据并行的一种新的方式. 微软之前Deepspeed框架中提出过三种级别的ZERO算法，FSDP可以看成是ZERO-3的实现。核心在于要把DDP中的all-reduce操作拆解为reduce-scatter和all-gather 操作。


## 3.2 DeepSpeed
### **DeepSpeed基础特性** 

**1. DeepSpeed Sparse Attention**： 用6倍速度执行10倍长的序列： DeepSpeed提供了稀疏 attention kernel 一种工具性技术，可支持长序列的模型输入，包括文本输入，图像输入和语音输入。与经典的稠密 Transformer 相比，它支持的输入序列长一个数量级，并在保持相当的精度下获得最高 6 倍的执行速度提升。它还比最新的稀疏实现快 1.5–3 倍。此外，我们的稀疏 kernel 灵活支持稀疏格式，使用户能够通过自定义稀疏结构进行创新。  

**2 比特 Adam 减少 5 倍通信量**： Adam 是一个在大规模深度学习模型训练场景下的有效的（也许是最广为应用的）优化器。然而，它与通信效率优化算法往往不兼容。因此，在跨设备进行分布式扩展时，通信开销可能成为瓶颈。我们推出了一种 1 比特 Adam 新算法，以及其高效实现。该算法最多可减少 5 倍通信量，同时实现了与Adam相似的收敛率。在通信受限的场景下，我们观察到分布式训练速度提升了 3.5 倍，这使得该算法可以扩展到不同类型的 GPU 群集和网络环境。

**3 mpi、gloo 和 nccl 等通信策略**： 
* mpi 是一种跨节点通信库，常用于 CPU 集群上的分布式训练；  
* gloo 是一种高性能的分布式训练框架，支持 CPU 和 GPU 上的分布式训练；  
* nccl 是 NVIDIA 提供的 GPU 专用通信库，被广泛应用于 GPU 上的分布式训练。  



混合精度训练：在 DeepSpeed 中，可以通过在配置文件中设置 “bf16.enabled”: true 来启用 BF16 混合精度训练，减少占用内存。混合精度训练是指在训练过程中同时使用FP16（半精度浮点数）和FP32（单精度浮点数）两种精度的技术。
在使用混合精度训练时，需要注意一些问题，例如梯度裁剪（Gradient Clipping）和学习率调整（Learning Rate Schedule）等。梯度裁剪可以防止梯度爆炸，学习率调整可以帮助模型更好地收敛。因此，在设置混合精度训练时，需要根据具体情况进行选择和配置。
	存储一份fp32的parameter，momentum和variance（统称model states）
	在forward开始之前，额外开辟一块存储空间，将fp32 parameter减半到fp16 parameter。
	正常做forward和backward，在此之间产生的activation和gradients，都用fp16进行存储。
	用fp16 gradients去更新fp32下的model states。
	当模型收敛后，fp32的parameter就是最终的参数输出。




## 3.3 Megatron-LM 

## 3.4 Megatron-DeepSpeed







