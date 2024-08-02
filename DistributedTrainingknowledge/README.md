![image](https://github.com/user-attachments/assets/fa5c53c8-99fc-4720-a874-6f9210efd9bb)
<h2 align="center"> <a href="">大模型分布式加速训练方法基础知识总结</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
  
[![zhihu](https://img.shields.io/badge/知乎-0084FF)](https://blog.csdn.net/zwqjoy/article/details/130732601)
[![CSDN](https://img.shields.io/badge/CSDE-yellow)](https://zhuanlan.zhihu.com/p/660567767)

</h5>


# 1. 分布式通信术语
* **1. Broadcast：** 广播，一对多
![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-1.png)  
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

## 3.2 DeepSpeed


## 3.3 Megatron-LM 

## 3.4 Megatron-DeepSpeed







