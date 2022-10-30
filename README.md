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
one_target_train: 运行create_data.py生成的训练数据存放在这里（此处对应单目标检测，如若需要多个目标检测，建议新建文件夹存放生成的训练数据，命名只需要将“one”改为对应目标的数目）  
test：存放测试图片  

## 训练流程  
### 1、数据准备  
我们将自己制作目标检测的数据集，只需要将目标粘入背景图中，并保持目标位置即可。









