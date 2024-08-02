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
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-7.png)  


参考文献：[https://blog.csdn.net/cy413026/article/details/138618053](https://blog.csdn.net/cy413026/article/details/138618053)

# 2. 分布式通信架构

* **2.1.All-reduce**  
     All-reduce架构中仅用到 GPU 机器，这是因为其设计假定了每个节点都是同构节点。迭代过程中，GPU 独立计算模型参数的梯度，然后使用 All-reduce 通信聚合梯度。  
* **2.2 PS 架构**   
     PS 则包含 GPU worker 和 CPU server。迭代过程中，GPU worker 将梯度传输至 CPU server；后者将接收到的不同 workers 的梯度做聚合，然后执行 DNN 优化器（如 RMSProp 或 Adam 等）并将更新后的参数传输回 GPU workers。
  ![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-8.png)     


# 3. 分布式训练框架
## 3.1 Pytorch原生支持DDP, FSDP

* **DDP**: 传统的数据并行, 每一个GPU卡上保存整个model的参数/梯度/优化器状态, 然后对数据集切分为 N个shard分片给不同的GPU进行训练，计算完梯度后通过all-reduce通信来做梯度的融合。

* **FSDP**: 全切片数据并行(Fully Sharded Data Parallel，简称为FSDP)是数据并行的一种新的方式. 微软之前Deepspeed框架中提出过三种级别的ZERO算法，FSDP可以看成是ZERO-3的实现。核心在于要把DDP中的all-reduce操作拆解为reduce-scatter和all-gather 操作。


## 3.2 DeepSpeed
### **DeepSpeed基础特性** 

**1. DeepSpeed Sparse Attention**： 用6倍速度执行10倍长的序列： DeepSpeed提供了稀疏 attention kernel 一种工具性技术，可支持长序列的模型输入，包括文本输入，图像输入和语音输入。与经典的稠密 Transformer 相比，它支持的输入序列长一个数量级，并在保持相当的精度下获得最高 6 倍的执行速度提升。它还比最新的稀疏实现快 1.5–3 倍。此外，我们的稀疏 kernel 灵活支持稀疏格式，使用户能够通过自定义稀疏结构进行创新。  

**2. 比特 Adam 减少 5 倍通信量**： Adam 是一个在大规模深度学习模型训练场景下的有效的（也许是最广为应用的）优化器。然而，它与通信效率优化算法往往不兼容。因此，在跨设备进行分布式扩展时，通信开销可能成为瓶颈。我们推出了一种 1 比特 Adam 新算法，以及其高效实现。该算法最多可减少 5 倍通信量，同时实现了与Adam相似的收敛率。在通信受限的场景下，我们观察到分布式训练速度提升了 3.5 倍，这使得该算法可以扩展到不同类型的 GPU 群集和网络环境。

**3. mpi、gloo 和 nccl 等通信策略**： 
```
* mpi 是一种跨节点通信库，常用于 CPU 集群上的分布式训练；  
* gloo 是一种高性能的分布式训练框架，支持 CPU 和 GPU 上的分布式训练；  
* nccl 是 NVIDIA 提供的 GPU 专用通信库，被广泛应用于 GPU 上的分布式训练。  
```

**4. 混合精度训练**：  
在 DeepSpeed 中，可以通过在配置文件中设置 “bf16.enabled”: true 来启用 BF16 混合精度训练，减少占用内存。混合精度训练是指在训练过程中同时使用FP16（半精度浮点数）和FP32（单精度浮点数）两种精度的技术。  
在使用混合精度训练时，需要注意一些问题，例如梯度裁剪（Gradient Clipping）和学习率调整（Learning Rate Schedule）等。梯度裁剪可以防止梯度爆炸，学习率调整可以帮助模型更好地收敛。因此，在设置混合精度训练时，需要根据具体情况进行选择和配置。  

![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-9.png) 

```
* 存储一份fp32的parameter，momentum和variance（统称model states）
* 在forward开始之前，额外开辟一块存储空间，将fp32 parameter减半到fp16 parameter。
* 正常做forward和backward，在此之间产生的activation和gradients，都用fp16进行存储。
* 用fp16 gradients去更新fp32下的model states。
* 当模型收敛后，fp32的parameter就是最终的参数输出。
```
注：W=fp16(参数 2Ψ)，G=fp16(梯度 2Ψ)，O=fp32(优化器状态 4Ψ+4Ψ+4Ψ=12Ψ)，假设模型参数在INT8下模型的参数大小为Ψ.

### **DeepSpeed-ZeRO** 

**0. ZeRO-0**: 禁用所有类型的分片，仅使用 DeepSpeed 作为 DDP (Distributed Data Parallel) (计算完梯度需要All-Reduce， 先聚合到GPU0，在广播， 通信量2Φ)  

**1. ZeRO-1**:  分割Optimizer states (优化器状态包括一份fp32的模型参数副本、 Adam优化器的两个参数<momentum, variance>),  不同优化器参数数量不一样， SGD只有Momentum. 
```
* 操  作：优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数。通信容量与数据并行性相同, 但可以减少了4倍的显存；
* 优化前：ZeRO-1采用先对梯度All-Reduce(所有)，通信量为2Φ；在对参数All-Gather（部分），通信量为2Φ；所以总的通信量为3Φ；
* 优化后：先对梯度Reduce Scatter（部分）， 通信量为1Φ； 在对参数All-Gather (部分)，通信量为1Φ； 总的通信量为2Φ；
```
（注：ZeRO-1前期通信量为3Φ，后期进行了代码优化通信量为2Φ）

![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-10.png) 

具体流程如下：  
- (1) 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，各得一份梯度;  
- (2) 对梯度做一次AllReduce，得到完整的梯度G，产生单卡通讯量 2Φ 。为了表达简明，这里通讯量我们就不再换算成byte了，而直接根据参数量来计算;   
- (3) 得到完整梯度G，就可以对W做更新。我们知道W的更新由Optimizer states;      
- (4) 此时，每块GPU上都有部分W没有完成更新（图中白色部分）。所以我们需要对W做一次All-Gather，从别的GPU上把更新好的部分W取回来。产生单卡通讯量 Φ ;   

在实操中，可以只对梯度做一次scatter-reduce，并用各自维护的optimizer去更新对应的W，然后再对W做all-gather使得每块卡上都有更新后的完整W，这样通讯量就是 2Φ 。因为论文定义stage1只有optimizer是切开的，意味着G和W都是完整的。所以对G做all-reduce（虽然拿回完整的G并没有意义），对W做all-gather，这样通讯量就是 3Φ 。deepspeed的某次代码更新是将stage1的通讯量从 3Φ 降至 2Φ ，可能也是基于此做了改进。


**2. ZeRO-2**: 分割Optimizer States与Gradients   
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-11.png)
```
* 操  作：每个memory，只保留它分配到的optimizer state所对应的梯度。
* 合理性：因为梯度和Optimizer是紧密联系在一起的。只知道梯度，不知道Optimizer state，是没有办法优化模型参数的。
* 收  益：8倍显存节约，先对梯度Scatter-Reduce（部分）， 通信量为1Φ， 在对参数All-Gather (部分)， 通信量为1Φ； 所以总的通信量为2Φ。
```

![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-12.png)

- (1) 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，算得一份完整的梯度（上图中绿色+白色）;  
- (2) 对梯度做一次Reduce-Scatter，保证每个GPU上所维持的那块梯度是聚合梯度。例如对GPU1，它负责维护G1，因此其他的GPU只需要把G1对应位置的梯度发给GPU1做加总就可;汇总完毕后，白色块对GPU无用，可以从显存中移除。单卡通讯量 Φ ;  
- (3) 每块GPU用自己对应的O和G去更新相应的W。更新完毕后，每块GPU维持了一块更新完毕的W。同理，对W做一次All-Gather，将别的GPU算好的W同步到自己这来, 单卡通讯量 Φ.  

 
**3. ZeRO-3**：分割Optimizer States、Gradients与Parameters；ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory（16Ψ/N）
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-13.png)
流程如下：
- (1) 每块GPU上只保存部分参数W。将一个batch的数据分成3份，每块GPU各吃一份；   
- (2) 做forward时，对W做一次All-Gather，取回分布在别的GPU上的W，得到一份完整的W，单卡通讯量 Φ 。forward做完，立刻把不是自己维护的W抛弃；   
- (3) 做backward时，对W做一次All-Gather，取回完整的W，单卡通讯量 Φ 。backward做完，立刻把不是自己维护的W抛弃；  
- (4) 做完backward，算得一份完整的梯度G，对G做一次Reduce-Scatter，从别的GPU上聚合自己维护的那部分梯度，单卡通讯量 Φ 。聚合操作结束后，立刻把不是自己维护的G抛弃。
- (5) 用自己维护的O和G，更新W。由于只维护部分W，因此无需再对W做任何AllReduce操作；  

**ZeRO-0 vs. ZeRO-1 vs. ZeRO-2 vs. ZeRO-3**：
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-14.png)

**4. ZeRO++**：对ZeRO-3进行了优化，3D并行化实现万亿参数模型训练；通信量减少4倍, 前向传播参数同步0.5Φ+反向传播梯度更新同步0.25Φ; 节点内部FP16—>INT8; 节点之间FP16->INT4!  
```
* 模型参数：模型参数每台服务器保存一份，服务器内部参数分布存储；
* 分层计算：每台服务器内部先更新参数，然后在服务器之间同步；
* 量化通信：通信数据量化为int8,然后反量化。
```

![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-15.png)

* **量化ZeRO权重通信（qwZ）**: 减少在all-gather期间的参数通信量，我们采用了对权重的量化，该方法在通信前将每个模型参数即时从FP16（两字节）压缩到INT8（一字节）的数据类型，并在通信后对权重进行反量化!

* **量化ZeRO梯度通信（qgZ）**: ZeRO 在后向计算完成之后需要一次 Reduce-Scatter 通信，如果直接将量化策略应用到 Reduce-Scatter 通信的话，包含超过一系列的量化和反量化操作（量化压缩后的数据以为所有 GPU 的平均数），这不仅需要较大的巨大的计算开销，还会带来额外的通信量； 为减少量化和反量化操作次数，首先对全部梯度量化，然后所有GPU进行一次All-to-All通信，最后执行反量化操作。

* **分层切片hpZ**：为了减少反向传播过程中权重上的 all-gather 通信开销，我们选择用 GPU 内存来换取通信。具体而言，我们并不是像在 ZeRO 中那样将整个模型权重分布在所有的机器中，而是在每台机器内维护一份完整的模型副本。尽管这会带来更高的内存开销，但它使我们能够用机器内的 all-gather/broadcast 替换昂贵的跨机器 all-gather/broadcast，由于机器内通信带宽更高，所以这个过程会快很多。

**5.ZeRO vs 模型并行** 

虽然ZeRO把参数W给切了，但ZeRO是模型并行的形式，数据并行的实质。模型并行，是指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行。即同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果。但对ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算。它是不同的输入X，完整的参数W，最终再做聚合。

**6.ZeRO-Offload** 
* forward和backward计算量高，因此和它们相关的部分，例如 参数W（fp16），activation，就全放入GPU  
* update的部分计算量低，因此和它相关的部分，全部放入CPU中。例如 optimizer states（fp32）和gradients(fp16)等  

### **DeepSpeed使用** 
```
* Zero（Zero Redundancy Optimizer，3D优化与卸载）：在deepspeed中通过zero_optimization.stage=0/1/2/3 设置
* 卸载通过zero_optimization.offload_optimizer.device设置
```
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-16.png)
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-17.png)



## 3.3 Megatron-LM 

## 3.4 Megatron-DeepSpeed







