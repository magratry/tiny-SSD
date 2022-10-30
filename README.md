# tiny-SSD
人工智能原理实验任务要求  
整理tiny-SSD代码  
1. 按照数据、模型、训练、测试、可视化、自定义函数等拆分为多个文件  
2. 提供readme文件，说明清楚环境配置、训练流程   
3. 提供简易测试代码及预训练模型   
4. 贴出检测效果  
5. 效果提升   
## 环境配置
CUDA Version = 10.1（可选）  
Python = 3.9  
PyTorch = 1.8.1  
matplotlib==3.5.3  
numpy==1.20.3  
opencv_python_headless==4.6.0.66  
pandas==1.3.4  
torch==1.8.1+cu101   
torchvision==0.9.1+cu101  
  
可以手动安装依赖，也可以直接运行以下代码   

`pip install -r requirements.txt`  
## 数据目录介绍  
```  
.
├─data
    ├─background
    │      000012-16347456122a06.jpg
    │	. . .
    │      191328-15136820086f91.jpg
    │      
    ├─one_target_train
    │  │  
    │  └─images
    │          
    ├─target
    │      0.png
    │      1.png
    │      
    ├─test
    │      1.jpg
    │      2.jpg
    │      
    └─two_target_train
        │  
        └─images  
```  
【data目录说明】  
background：将你自己准备的背景图片放在这里  
target：目标图片放在这里【注意：目标图片的命名按照0.png、1.png…来命名】  
one_target_train：生成的训练数据存放在这里（此处对应单目标检测，如若需要多个目标检测，建议新建文件夹存放生成的训练数据，命名只需要将“one”改为对应目标的数目）  
test：存放测试图片    


  
  
## 训练流程  
代码结构如下：  
1. create_data.py
2. load_data.py  
3. model.py
4. train.py
5. test.py
6. plot.py
7. util.py
  
其中，代码1，4，5为主要运行代码，其余为辅助代码。  
主要运行代码主函数开头都有一个变量：target_num，如果设置其为'one'，则对应一个目标；如果设置其为'two'，则对应两个目标。
  

  
  
### 1、数据准备  
我们会自己制作目标检测的数据集，目标将被粘入背景图中，并保存目标位置即可。  
运行create_data.py生成训练数据，target被粘在background文件夹中的每一张图片上，粘贴的位置随机。  
生成的新图片将存入one_target_train/images文件夹中，图片对应粘贴的位置将生成label.csv存入one_target_train文件夹中。    

`python create_train.py`   
  
### 2、模型训练  

 








