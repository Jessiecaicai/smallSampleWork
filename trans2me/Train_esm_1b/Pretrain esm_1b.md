# Pretrain esm_1b

数据集：Uniref50

数据放在`data/`文件夹下，模型放在`model/`文件夹下，运行程序为DDP_train_1b.py

直接运行`DDP_train_1b.py`即可（多机多卡）

为了减少模型参数，这里使用的模型参数为：

```
24 layers, 
768 embedding_size,
2560 ffn_embedding_size
16 heads
```

模型效果：（**左：自己训练效果**，**右：esm-1b的模型效果**）

![image-20210912200840297](D:\桌面Desktop\预训练_esm_1b\Contact prediction.assets\image-20210912200840297.png)<img src="D:\桌面Desktop\预训练_esm_1b\Contact prediction.assets\image-20210905143310938.png" alt="image-20210905143310938" style="zoom:50%;" />

（右边模型保存时的loss是1.90~1.99，并且loss还在持续下降，供参考）

***
果子哥，你竟然在工作
在生活

吵什么
你竟然在工作 晚上十点
晚上吃完海底捞居然还回去
明天还买买提呢
#胖死你们
下班了
