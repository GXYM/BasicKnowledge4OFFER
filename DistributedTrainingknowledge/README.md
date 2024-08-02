![image](https://github.com/user-attachments/assets/fa5c53c8-99fc-4720-a874-6f9210efd9bb)
<h2 align="center"> <a href="">大模型分布式加速训练方法基础知识总结</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
  
[![zhihu](https://img.shields.io/badge/知乎-0084FF)](https://blog.csdn.net/zwqjoy/article/details/130732601)
[![CSDN](https://img.shields.io/badge/CSDE-yellow)](https://zhuanlan.zhihu.com/p/660567767)

</h5>


# 1. 分布式通信术语
* **1. Broadcast：** 广播，一对多
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img1.png width=50% />
</div>

* **2. Reduce：** 各设备上相同位置的元素进行加和，并将结果呈现在一个设备上
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img2.png width=50% />
</div> 

* **3. All Reduce：** 相当于Reduce之后再来了一个Broadcast
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img3.png width=50% />
</div>

* **4. Gather：** Gather的中文叫做收集（即把东西放到一起，并不做运算），与Reduce不同的地方是，Gather只是将数据汇总到一起，而Reduce需要“按照指定的映射函数进行运算”
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img4.png width=50% />
</div>

* **5. All Gather：** 多对多广播 
* **6. Scatter：** 离散，扩散；即将一个机器上的不同数据分别给到不同机器。而广播的含义是将一个机器上的数据全部传输给其他机器
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img5.png width=50% />
</div>

* **7. Reduce Scatter：** 先广播在加和；Reduce_scatter最终呈现效果为：每个GPU上有一块完整加和后的数据。他和All reduce的区别在于，All reduce是所有完整加和的数据。
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img6.png width=50% />
</div>


<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img7.png width=50% />
</div>

参考文献：[https://blog.csdn.net/cy413026/article/details/138618053](https://blog.csdn.net/cy413026/article/details/138618053)

# 2. 分布式通信架构

* **2.1.All-reduce**  
     All-reduce架构中仅用到 GPU 机器，这是因为其设计假定了每个节点都是同构节点。迭代过程中，GPU 独立计算模型参数的梯度，然后使用 All-reduce 通信聚合梯度。  
* **2.2 PS 架构**   
     PS 则包含 GPU worker 和 CPU server。迭代过程中，GPU worker 将梯度传输至 CPU server；后者将接收到的不同 workers 的梯度做聚合，然后执行 DNN 优化器（如 RMSProp 或 Adam 等）并将更新后的参数传输回 GPU workers。

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img8.png width=50% />
</div>


# 3. 分布式训练框架
## 3.1 Pytorch原生支持DDP, FSDP

* **DDP**: 传统的数据并行, 每一个GPU卡上保存整个model的参数/梯度/优化器状态, 然后对数据集切分为 N个shard分片给不同的GPU进行训练，计算完梯度后通过all-reduce通信来做梯度的融合。

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img9.png width=50% />
</div>

* **FSDP**: 全切片数据并行(Fully Sharded Data Parallel，简称为FSDP)是数据并行的一种新的方式. 微软之前Deepspeed框架中提出过三种级别的ZERO算法，FSDP可以看成是ZERO-3的实现。核心在于要把DDP中的all-reduce操作拆解为reduce-scatter和all-gather 操作。


## 3.2 DeepSpeed
### **3.2.1 DeepSpeed基础特性** 

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

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img10.png width=50% />
</div>

```
* 存储一份fp32的parameter，momentum和variance（统称model states）
* 在forward开始之前，额外开辟一块存储空间，将fp32 parameter减半到fp16 parameter。
* 正常做forward和backward，在此之间产生的activation和gradients，都用fp16进行存储。
* 用fp16 gradients去更新fp32下的model states。
* 当模型收敛后，fp32的parameter就是最终的参数输出。
```
注：W=fp16(参数 2Ψ)，G=fp16(梯度 2Ψ)，O=fp32(优化器状态 4Ψ+4Ψ+4Ψ=12Ψ)，假设模型参数在INT8下模型的参数大小为Ψ.

```
Adam优化下的optimizer states只在最终做update时才用到
数据并行中，gradients只在最后做AllReduce和updates时才用到
参数W只在做forward和backward的那一刻才用到
```


### **3.2.2 DeepSpeed-ZeRO** 

**0. ZeRO-0**: 禁用所有类型的分片，仅使用 DeepSpeed 作为 DDP (Distributed Data Parallel) (计算完梯度需要All-Reduce， 先聚合到GPU0，在广播， 通信量2Φ)  

**1. ZeRO-1**:  分割Optimizer states (优化器状态包括一份fp32的模型参数副本、 Adam优化器的两个参数<momentum, variance>),  不同优化器参数数量不一样， SGD只有Momentum. 
```
* 操  作：优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数。通信容量与数据并行性相同, 但可以减少了4倍的显存；
* 优化前：ZeRO-1采用先对梯度All-Reduce(所有)，通信量为2Φ；在对参数All-Gather（部分），通信量为2Φ；所以总的通信量为3Φ；
* 优化后：先对梯度Reduce Scatter（部分）， 通信量为1Φ； 在对参数All-Gather (部分)，通信量为1Φ； 总的通信量为2Φ；
```
（注：ZeRO-1前期通信量为3Φ，后期进行了代码优化通信量为2Φ）

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img11.png width=50% />
</div>

具体流程如下：  
- (1) 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，各得一份梯度;  
- (2) 对梯度做一次AllReduce，得到完整的梯度G，产生单卡通讯量 2Φ 。为了表达简明，这里通讯量我们就不再换算成byte了，而直接根据参数量来计算;   
- (3) 得到完整梯度G，就可以对W做更新。我们知道W的更新由Optimizer states;      
- (4) 此时，每块GPU上都有部分W没有完成更新（图中白色部分）。所以我们需要对W做一次All-Gather，从别的GPU上把更新好的部分W取回来。产生单卡通讯量 Φ ;   

在实操中，可以只对梯度做一次scatter-reduce，并用各自维护的optimizer去更新对应的W，然后再对W做all-gather使得每块卡上都有更新后的完整W，这样通讯量就是 2Φ 。因为论文定义stage1只有optimizer是切开的，意味着G和W都是完整的。所以对G做all-reduce（虽然拿回完整的G并没有意义），对W做all-gather，这样通讯量就是 3Φ 。deepspeed的某次代码更新是将stage1的通讯量从 3Φ 降至 2Φ ，可能也是基于此做了改进。


**2. ZeRO-2**: 分割Optimizer States与Gradients   
```
* 操  作：每个memory，只保留它分配到的optimizer state所对应的梯度。
* 合理性：因为梯度和Optimizer是紧密联系在一起的。只知道梯度，不知道Optimizer state，是没有办法优化模型参数的。
* 收  益：8倍显存节约，先对梯度Scatter-Reduce（部分）， 通信量为1Φ， 在对参数All-Gather (部分)， 通信量为1Φ； 所以总的通信量为2Φ。
```
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img12.png width=50% />
</div>  

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img13.png width=50% />
</div>

- (1) 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，算得一份完整的梯度（上图中绿色+白色）;  
- (2) 对梯度做一次Reduce-Scatter，保证每个GPU上所维持的那块梯度是聚合梯度。例如对GPU1，它负责维护G1，因此其他的GPU只需要把G1对应位置的梯度发给GPU1做加总就可;汇总完毕后，白色块对GPU无用，可以从显存中移除。单卡通讯量 Φ ;  
- (3) 每块GPU用自己对应的O和G去更新相应的W。更新完毕后，每块GPU维持了一块更新完毕的W。同理，对W做一次All-Gather，将别的GPU算好的W同步到自己这来, 单卡通讯量 Φ.  

 
**3. ZeRO-3**：分割Optimizer States、Gradients与Parameters；ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory（16Ψ/N）
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img14.png width=50% />
</div>

流程如下：  
- (1) 每块GPU上只保存部分参数W。将一个batch的数据分成3份，每块GPU各吃一份；     
- (2) 做forward时，对W做一次All-Gather，取回分布在别的GPU上的W，得到一份完整的W，单卡通讯量 Φ 。forward做完，立刻把不是自己维护的W抛弃；   
- (3) 做backward时，对W做一次All-Gather，取回完整的W，单卡通讯量 Φ 。backward做完，立刻把不是自己维护的W抛弃；  
- (4) 做完backward，算得一份完整的梯度G，对G做一次Reduce-Scatter，从别的GPU上聚合自己维护的那部分梯度，单卡通讯量 Φ 。聚合操作结束后，立刻把不是自己维护的G抛弃。
- (5) 用自己维护的O和G，更新W。由于只维护部分W，因此无需再对W做任何AllReduce操作；  

**ZeRO-0 vs. ZeRO-1 vs. ZeRO-2 vs. ZeRO-3**：
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img15.png width=50% />
</div>

**4. ZeRO++**：对ZeRO-3进行了优化，3D并行化实现万亿参数模型训练；通信量减少4倍, 前向传播参数同步0.5Φ+反向传播梯度更新同步0.25Φ; 节点内部FP16—>INT8; 节点之间FP16->INT4!  
```
* 模型参数：模型参数每台服务器保存一份，服务器内部参数分布存储；
* 分层计算：每台服务器内部先更新参数，然后在服务器之间同步；
* 量化通信：通信数据量化为int8,然后反量化。
```

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img16.png width=50% />
</div>

* **量化ZeRO权重通信（qwZ）**: 减少在all-gather期间的参数通信量，我们采用了对权重的量化，该方法在通信前将每个模型参数即时从FP16（两字节）压缩到INT8（一字节）的数据类型，并在通信后对权重进行反量化!

* **量化ZeRO梯度通信（qgZ）**: ZeRO 在后向计算完成之后需要一次 Reduce-Scatter 通信，如果直接将量化策略应用到 Reduce-Scatter 通信的话，包含超过一系列的量化和反量化操作（量化压缩后的数据以为所有 GPU 的平均数），这不仅需要较大的巨大的计算开销，还会带来额外的通信量； 为减少量化和反量化操作次数，首先对全部梯度量化，然后所有GPU进行一次All-to-All通信，最后执行反量化操作。

* **分层切片hpZ**：为了减少反向传播过程中权重上的 all-gather 通信开销，我们选择用 GPU 内存来换取通信。具体而言，我们并不是像在 ZeRO 中那样将整个模型权重分布在所有的机器中，而是在每台机器内维护一份完整的模型副本。尽管这会带来更高的内存开销，但它使我们能够用机器内的 all-gather/broadcast 替换昂贵的跨机器 all-gather/broadcast，由于机器内通信带宽更高，所以这个过程会快很多。

**5.ZeRO vs 模型并行** 

虽然ZeRO把参数W给切了，但ZeRO是模型并行的形式，数据并行的实质。模型并行，是指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行。即同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果。但对ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算。它是不同的输入X，完整的参数W，最终再做聚合。

**6.ZeRO-Offload** 
* forward和backward计算量高，因此和它们相关的部分，例如 参数W（fp16），activation，就全放入GPU  
* update的部分计算量低，因此和它相关的部分，全部放入CPU中。例如 optimizer states（fp32）和gradients(fp16)等  

### **3.2.3 DeepSpeed使用** 
```
* Zero（Zero Redundancy Optimizer，3D优化与卸载）：在deepspeed中通过zero_optimization.stage=0/1/2/3 设置
* 卸载通过zero_optimization.offload_optimizer.device设置
```
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img17.png width=50% />
</div>


### 3.2.4 显存占用分析  
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img18.png width=50% />
</div>

混合精度训练，同时存在fp16和fp32两种格式的数值，其中模型参数、模型梯度都是fp16，此外还有fp32的模型参数，如果优化器是Adam，则还有fp32的momentum和variance。
总的来说，模型训练时显存主要分为两部分。

* 存储主要分为两大块：**Model States**和**Residual States**:
```
* Model States指和模型本身息息相关的，必须存储的内容，具体包括：
* optimizer states：Adam优化算法中的momentum和variance
* gradients：模型梯度
* parameters：模型参数W
* Residual States指并非模型必须的，但在训练过程中会额外产生的内容，具体包括：
* activation：激活值。在流水线并行中我们曾详细介绍过。在backward过程中使用链式法则计算梯度时会用到。有了它算梯度会更快，但它不是必须存储的，因为可以通过重新做Forward来算它。
* temporary buffers: 临时存储。例如把梯度发送到某块GPU上做加总聚合时产生的存储。
* unusable fragment memory：碎片化的存储空间。虽然总存储空间是够的，但是如果取不到连续的存储空间，相关的请求也会被fail掉。对这类空间浪费可以通过内存整理来解决。
```
模型在训练过程中需要储存自身的参数和梯度（注意这里还不是Adam最后算出来的参数更新量，只是根据loss反向传播得到的原始梯度），这便需要 2Ψ+2Ψ 的内存，同时混合精度fp32训练时，Adam需要一份fp32大小的模型拷贝，momentum和variance去储存模型的优化器状态，这需要 4Ψ+4Ψ+4Ψ ，最终我们需要 16Ψ𝐵 的内存用于训练，即对于一个GPT-2模型，我们训练时需要24GB的内存，对比一张V100的显存为32GB

**ZeRO-DP主要是优化第一部分的显存占用，所以这里主要介绍第一部分的显存**
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img19.png width=50% />
</div>

* **1.将权重转换为FP16**：在这一步中，神经网络的权重（或参数）最初是FP32格式，被转换为低精度的FP16格式。这减少了内存的占用，并允许更快的计算，因为FP16操作需要更少的内存，并且可以被硬件更快地处理。计算梯度：神经网络的前向和后向通道是使用较低精度的FP16权重进行的。这一步计算损失函数相对于网络权重的梯度（部分导数），在优化过程中用于更新权重。  
* **2.将梯度转换为FP32**：在FP16中计算梯度后，它们被转换回高精度的FP32格式。这种转换对于保持数值稳定性和避免使用低精度算术时可能出现的梯度消失或爆炸等问题至关重要。乘以学习率和更新权重：现在是FP32格式，梯度被乘以学习率（一个标量值，决定了优化过程中的步长）。乘积被用来更新原始FP32神经网络权重。学习率有助于控制优化过程的收敛性，对于实现良好的性能至关重要。
    
**模型状态（model states）：假设模型的参数量是 Ψ ，使用Adam为优化器进行混合精度训练**  
* a.由于模型的参数和梯度使用float16，所以显存消耗分别为 2Ψ 和 2Ψ 。  
* b. Adam会维护一个float32的模型备份副本，消耗 4Ψ 显存。Adam优化器本身会为模型的每个参数维护两个float32的辅助变量（fp32的momentum和fp32的variance），所以显存消耗占用为 4Ψ+4Ψ 。  
总的来说，模型会消耗 2Ψ+2Ψ=4Ψ ，Adam优化器这消耗 4Ψ+4Ψ+4Ψ=12Ψ 。最终的总消耗为 4Ψ+12Ψ=16Ψ 。  

混合精度（fp16/32）训练，模型参数和激活值是以fp16的格式进行保存，前向和反向传播中也使用fp16的权重和激活值进行计算。然而为了在反向传播结束时有效地计算和参数更新，保证梯度更新的正确（混合精度训练fp16会有一个大的累积舍入误差，例如大的浮点数+小的浮点数，会体现不出来），通常会同时拷贝一份fp32的参数权重和优化器状态(包括动量估计+梯度方差)。  
即混合精度使用Adam优化算法，需要保存以下状态信息：
* Fp16的参数、梯度（2 +2 内存消耗）
* Fp32的参数、动量、方差（4 +4 +4 内存消耗）

**以LLaMA 7B 模型为例:**

1. 模型参数P所需的内存是(参数量*每个参数的内存)：  
   混合精度（fp16/32）训练: 存储fp16精度+fp32精度 = 14GB +28GB = 42GB  
2. 梯度G所需的内存（计算方式同模型参数一样，参数量*每个参数的内存）：
   混合精度（fp16/32）训练: 只存储fp16精度 = 14GB  
3. 以Adam为例，它需要存储两部分的优化器状态：time averaged momentum(动量估计)和variance of the gradients(梯度方差)。  
   混合精度（fp16/32）训练: 存储fp32精度 = 56 G
model states所需的内存，混合精度（fp16/32）训练: 42+14+56 = 112GB

**Residual States**
除了模型状态之外的显存占用，包括激活值（activation）、各种临时缓冲区（buffer）以及无法使用的显存碎片（fragmentation）。
显然，激活在训练中也会消耗大量的显存。尽管激活的显存占用已经显著减少，但是对于更大的模型来说，激活所占用的显存也会非常大。  

例如，对于100B参数量的GPT模型且batch size为32，即使用来activation checkpointing，显存占用也需要60GB。
* **Activation checkpointing**: 前向计算时，只保留部分算子的激活值。 反向更新时，需要其他算子的激活值时，再重新对其进行前向计算，得到其激活值。分为两种方式：
* **full checkpoint**: 对所有算子都进行Activation checkpointing，等于走了两遍完整的前向计算，虽然将内存消耗减小到平方根的量级，即从60GB->8GB； 但是会带来36%的重新计算开销。
* **Selective checkpointing**: 只对那些计算时间小，占显存多的op（如attention部分的op）进行Activation checkpointing。重新计算开销从36% -> 4%
* **临时缓存区(Temporary buffers)**: 对于大模型，用于存储中间结果的临时buffer也会消耗大量显存。例如在all-reduce时，需要一个平坦的buffer来融合所有的梯度，从而改善吞吐量。例如，跨设备的all-reduce操作会随着消息的增大而增加。虽然，梯度本文是fp16的张量，但是有些操作中可能需要融合的buffer为fp32。当模型尺寸很大时，临时的buffer也不小。例如，对于1.5B参数的模型，一个fp32的buffer需要6GB的显存。
* **显存碎片**: 即使在有足够显存的情况下，也可能会导致Out of Memory，这是由于显存碎片导致的。在进程发出显存请求时，如果没有连续的显存来满足请求，即使总的显存仍然足够，该请求也会失败。当训练非常大的模型时，可以观察到明显的显存碎片。极端情况下，可能会导致30%的显存碎片。


**使用DeepSpeed ZeRO策略训练LLama-7B参数模型，8卡机器，混合精度训练，显存占用情况如下**：  
计算每张卡的总显存占用情况：
* 模型参数(2W)：模型参数存储占用约14GB（FP16）(每卡显存=14GB/8=1.75GB    zero3)
* 梯度(2W)： 梯度存储占用约14GB（FP16).  (每卡显存=14GB/8=1.75GB    zero2,  zero3)
* 优化器状态(12W)： 每卡优化器状态占用10.5GB。[优化器状态总共占用84GB（7GB * 12），在8张卡上进行分片，
* 每张卡分担的优化器状态显存为：每卡优化器状态显存=84GB/8=10.5GB, 每卡 优化器状态显存=10.5G （zero-1, zero-2, zero3）

**激活值显存占用计算步骤**:
```
模型架构：假设LLaMA 7B模型具有32层，每层有4096个隐藏单元。
batch size和序列长度：假设batch size为1，序列长度为1024。
激活值大小：激活值的大小取决于batch size、序列长度和隐藏层维度。
```

**计算激活值显存占用如下**：
* 单层激活值计算: 每层激活值占用的显存大小为：batch size * 序列长度 * 隐藏层维度全模型激活值计算
* 总激活值显存占用为：模型层数 * 单层激活值大小
  
**假设使用FP16（每个值2字节），具体计算如下**：
单层激活值大小：
```
激活值大小 = 1（batch size） * 1024（序列长度） * 4096（隐藏层维度）* 2字节  
单层激活值大小 = 1 * 1024 * 4096 * 2 = 8,388,608 字节 ≈ 8MB
```
全模型激活值大小：
```
总激活值大小 = 32（层数） * 8MB（单层激活值大小）= 256M
```
在上述假设条件下，使用DeepSpeed ZeRO-1策略训练LLaMA 7B模型，混合精度训练时，激活值的显存占用大约为 256M。这个计算是基于32层，每层4096个隐藏单元，batch size为1，序列长度为1024的假设条件下进行的。实际显存占用会因具体的模型架构和训练配置而有所不同。 

总显存占用：
```
每张卡大约需要： 模型参数 + 梯度 + 分片优化器状态 + 激活显存。
```

**zero1:**  
估算：14GB（模型参数FP16） + 14GB（梯度FP16） + 10.5GB（分片优化器状态） + 0.25 GB（激活显存）。
每卡总显存占用约 14GB + 14GB + 10.5GB + 0.25 GB = 38.75G

**zero2:**  
估算：14GB（模型参数FP16） + 1.75GB（梯度FP16) + 10.5GB（分片优化器状态） + 0.25 GB（激活显存）。
每卡总显存占用约 14GB + 1.75GB + 10.5GB + 0.25 GB = 26.5G

**zero3:**  
估算：1.75GB（模型参数FP16） + 1.75GB（梯度FP16） + 10.5GB（分片优化器状态） + 0.25 GB（激活显存）。
每卡总显存占用约 1.75GB + 1.75GB + 10.5GB + 0.25 GB = 14.25G

### 3.2.6 总结
微软开发ZeRO是为了克服数据并行性和模型并行性的限制，同时实现两者的优点。
* ZeRO通过在数据并行进程中划分模型状态（参数，梯度和优化器状态），而不是复制它们，从而消除了数据并行进程中的内存冗余。它在训练期间使用动态通信计划，以在分布式设备之间共享必要的状态，以保持计算粒度和数据并行性的通信量。  
* ZeRO驱动的数据并行性，它允许每个设备的内存使用量随数据并行性的程度线性扩展，并产生与数据并行性相似的通信量。 ZeRO支持的数据并行性可以适合任意大小的模型，只要聚合的设备内存足够大以共享模型状态即可。  


## 3.3 Megatron-LM 
是由 NVIDIA 应用深度学习研究团队开发的大型、强大的 transformer 模型框架; 论文：https://arxiv.org/pdf/1909.08053
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img20.png width=50% />
</div>

布式环境初始化，即按照DP/TP/PP对进程进行分组，并为每个进程指定GPU。例如：CodeGeeX在预训练中采用的是8头TP（同一个node内的8张卡做TP，8张卡组成一个完整的模型），192头DP（192个node间做DP），一共1536块GPU进行。  

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img21.png width=50% />
</div>

### 3.3.1 张量并行(Tensor Parallelism，算模型并行的一种)  
每个张量都被分成多个块，因此张量的每个分片都位于其指定的 GPU 上，而不是让整个张量驻留在单个 GPU 上。在处理过程中，每个分片在不同的 GPU 上分别并行处理，结果在步骤结束时同步。这就是所谓的水平并行，因为是做的水平拆分

<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img22.png width=50% />
</div>

### 3.3.2 流水线并行Pipeline Parallelism(模型并行的另一种)
朴素流水线并行 (naive PP) 是将模型各层分组分布在多个 GPU 上，并简单地将数据从 GPU 移动到 GPU，就好像它是一个大型复合 GPU 一样。该机制相对简单 - 将所需层用 .to() 方法绑到相应设备，现在只要数据进出这些层，这些层就会将数据切换到与该层相同的设备，其余部分保持不变  
这其实就是垂直模型并行(类似画大多数模型的拓扑图，垂直切分模型各层的)，例如，下图显示一个 8 层模型  
<div align=center>
  <img src=https://github.com/GXYM/BasicKnowledge4OFFER/blob/main/DistributedTrainingknowledge/dkimgs/img23.png width=50% />
</div>

## 3.4 Megatron-DeepSpeed

DeepSpeed团队通过将“下面第一项与后面三项相结合”，开发了一种基于3D并行的实现，这就是Megatron-Deepspeed，它使得千亿级参数量以上的大规模语言模型比如BLOOM的分布式训练变得更简单、高效和有效.  

**(1) Megatron-LM中的张量并行(Tensor Parallelism，可以理解为模型并行的一种)**  
每个张量都被分成多个块，因此张量的每个分片都位于其指定的 GPU 上，而不是让整个张量驻留在单个 GPU 上。在处理过程中，每个分片在不同的 GPU 上分别并行处理，结果在步骤结束时同步。这就是所谓的水平并行，因为是做的水平拆分

**(2) 零冗余优化器 (Zero Redundancy Optimizer，简称ZeRO，是微软DeepSpeed库的核心)**  
也执行与 TP 相类似的张量分片，但整个张量会及时重建以进行前向或反向计算，因此不需要修改模型。它还支持各种卸载技术以补偿有限的 GPU 内存


**(3) 数据并行(Data Parallelism)**  
相同的设置和模型被复制多份，每份每次都被馈送不同的一份数据。处理是并行完成的，所有份在每个训练步结束时同步

**(4) 管道并行(也称流水线并行，Pipeline Parallelism)** 
模型在多个 GPU 上垂直 (即按层) 拆分，因此只有一个或多个模型层放置在单个 GPU 上。每个 GPU 并行处理流水线的不同阶段，并处理 batch 的一部分数据




# 4. 参考文献
1. https://blog.csdn.net/cy413026/article/details/138618053
2. https://blog.csdn.net/zwqjoy/article/details/130732601
3. https://zhuanlan.zhihu.com/p/634377071
4. https://zhuanlan.zhihu.com/p/709639748
5. [《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》](https://arxiv.org/pdf/1909.08053)










