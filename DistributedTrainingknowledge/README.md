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
  ![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-5.png)  
* **6. Scatter：** 离散，扩散；即将一个机器上的不同数据分别给到不同机器。而广播的含义是将一个机器上的数据全部传输给其他机器
   ![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-6.png)  
* **7. Reduce Scatter：** 先广播在加和；Reduce_scatter最终呈现效果为：每个GPU上有一块完整加和后的数据。他和All reduce的区别在于，All reduce是所有完整加和的数据。
  ![](https://github.com/GXYM/BasicKnowledge4OFFER/tree/main/DistributedTrainingknowledge/DTK-imgs/img-7.png)   



## 2. Prepare Dataset   
1. Download the corresponding video data through the [download_scripts.](https://github.com/GXYM/TextBPN/blob/main/vis/1.png)  we have collected.

## 3. Environment
 * 1. We provide environment dependencies, first install [requirements_all.txt](https://github.com/GXYM/STGT/blob/main/requirements_all.txt), then install [requirements_all-1.txt](https://github.com/GXYM/STGT/blob/main/requirements_all-1.txt)
```
pip install -r requirements_all.txt
pip install -r requirements_all-1.txt
```
 *  2. You can also run the [pip_install.sh](https://github.com/GXYM/STGT/blob/main/pip_install.sh) directly
```
sh pip_install.sh
```

## 4. DownLoad Models
The model link have been shared herer.

|         Data    |  W10M+VIDAL4M-256|W10M+VIDAL4M-1024 | W10M+VIDAL7M-256 |Extracted Code|
|:------------------:	|:-----------:  |:-----------:	|:-------:|:-------:|
| Pre-training Models |  [checkpoint_w10m_v4m_256.pth](https://pan.baidu.com/s/1eB7-ViWPf1l9gdDhkYXFsQ) | [checkpoint_w10m_v4m_1024.pth](https://pan.baidu.com/s/1jP9rLMyyZ2mteD7kwu1irw) 	| [checkpoint_w10m_v7m_256.pth](https://pan.baidu.com/s/1afl0BzUzhkbn_P3eSIF8TQ) |gxym|

|         Dataset   |  model-1| model-2 |Extracted Code|
|:------------------:	|:-----------:	|:-------:|:-------:|
| didemo_ret| [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/1yezEntt8w0rQVG99jy12JA)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/1yezEntt8w0rQVG99jy12JA)|gxym|
| lsmdc_ret | [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/19zdiscvvoeeJjZ9v5zMIrg)| [checkpoint_best_w10m_v4m_256.pth](https://pan.baidu.com/s/19zdiscvvoeeJjZ9v5zMIrg)|gxym|
| msrvtt_reT| [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/1NC7vGWW5hkwP8V72Fwpxig)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/1NC7vGWW5hkwP8V72Fwpxig)|gxym|
| msvd_ret  | [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/18QUC_gUMleswxymVKR-zSA)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/18QUC_gUMleswxymVKR-zSA)|gxym|
  
CLIP VIT pretrained models are [here](https://pan.baidu.com/s/13ITPJF2HFjep06BosK7E4w)

## 5.Eval and Testing

You can find the corresponding evaluation script [here](https://github.com/GXYM/STGT/tree/main/run_scripts/stgt/eval), configure the model path, and run it directly.  
```
DiDemo:  eval_didemo_ret_pretrain_vig.sh
LSMDC:   eval_lsmdc_ret_pretrain_vig.sh
MSRVTT:  eval_msrvtt_ret_pretrain_vig.sh
MSVD:    eval_msvd_ret_pretrain_vig.sh
```

We also provide an [ALPRO](https://github.com/GXYM/STGT/tree/main/run_scripts/alpro) evaluation scripts, and you can download its model for comparative testing.  

NOTE: Due to the desensitization process of the code, we cannot guarantee that there are no bugs in the code, but we will promptly fix these bugs.
 ## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{zhang2024video,
  title={Video-Language Alignment Pre-training via Spatio-Temporal Graph Transformer},
  author={Zhang, Shi-Xue and Wang, Hongfa and Zhu, Xiaobin and Gu, Weibo and Zhang, Tianjin and Yang, Chun and Liu, Wei and Yin, Xu-Cheng},
  journal={arXiv preprint arXiv:2407.11677},
  year={2024}
}
```

<!---->
## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=GXYM/STGT&type=Date)](https://star-history.com/#GXYM/STGT&Date)




