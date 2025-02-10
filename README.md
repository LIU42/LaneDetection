# Lane Detection

**基于网格级图像分割的车道线检测模型。**

## 项目简介

本项目设计并实现了一种基于网格级图像分割的车道线检测模型。

### 模型结构

本项目结合基于图像分割的车道线检测方法（例如 [LaneNet](https://arxiv.org/abs/1802.05591)）和基于行选择的车道线检测方法（例如 [Ultra Fast Lane Detection](https://arxiv.org/abs/2004.11757)）的特点，设计了一种网格级图像分割的车道线检测方法。与常规的像素级图像分割不同，模型对图像进行网格划分，预测每个网格中存在车道线的概率，可以在一定程度上减少计算量。

具体来说，本项目将尺寸为 640 x 360 的输入图像均匀划分为 80 x 45 个网格进行预测，即网格的宽高均为 8 个像素。模型结构总体上类似 U-Net 的结构，采用预训练的 ResNet 作为特征编码器，使用反卷积进行上采样操作，且每层的上采样结果图与对应尺寸的编码器输出特征图进行拼接融合，直到特征图的空间分辨率达到划分的网格尺寸。

此外为了进一步减少计算量，考虑到车道线不应该出现在图像的上部且图像上部的信息对车道线检测基本无用，本项目仅将图像的下部区域输入模型，同时将分割掩码图对应尺寸的下部区域作为标签。因此模型的输入数据尺寸为 (B, 3, 224, 640)，输出结果的数据尺寸为 (B, 1, 28, 80)。

### 训练策略

与大多数图像分割模型相同，本项目采用 Dice 损失函数训练模型。

模型采用 [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) 数据集进行训练，对原始的标签进行了处理，使之符合模型训练数据的要求（具体要求见下文），其中训练集 3626 张图像，测试集 2782 张图像，验证集 200 张图像。验证集图像从测试集中随机抽取得到，旨在判断模型的收敛性以及是否出现过拟合。

## 效果展示

以下是测试集中部分图像模型预测结果的可视化展示，其中点标注为预测结果中车道线所在网格的中心点。

![效果展示](assets/examples.jpg "效果展示")

## 性能评估

### 检测精度指标

本项目采用精确率 (Precision)，召回率 (Recall)，F1-score 以及预测结果图和实际分割掩码图的交并比 (IoU) 作为模型性能的检测精度评估指标，下面给出当前在 [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) 数据集下训练得到的最优模型检测精度指标：

| Precision | Recall | F1-score | IoU   |
| --------- | ------ | -------- | ----- |
| 0.787     | 0.784  | 0.786    | 0.658 |

## 使用说明

### 环境搭建

首先需要安装本项目依赖的各种库和工具包。

```bash
pip install -r requirements.txt
```

### 数据集准备

本项目的训练数据集格式如下，分为训练集、验证集和测试集，所有的图像文件需要调整尺寸至 640 x 360 并放入相应的 <u>images/</u> 目录下，分割掩码图像文件尺寸需为 80 x 45 并放入相应的 <u>labels/</u> 目录下。

```bash
datasets/
├── test/
│   ├── images/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── xxx.png
│   │   ├── xxx.png
│   │   └── ...
│   └── annotations.json
├── train/
│   ├── images/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── xxx.png
│   │   ├── xxx.png
│   │   └── ...
│   └── annotations.json
└── valid/
    ├── images/
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   └── ...
    ├── labels/
    │   ├── xxx.png
    │   ├── xxx.png
    │   └── ...
    └── annotations.json
```

其中 annotations.json 为所在数据集划分的索引文件。annotations.json 文件中的每一行 JSON 格式数据为一个样本标签对的信息，其格式及说明如下。

```json5
{
    "image": "xxx.jpg",    // "image" 为输入图像的文件名，对应 images/ 目录下的图像文件名。
    "label": "xxx.png"     // "label" 为分割掩码图像的文件名，对应 labels/ 目录下的图像文件名。
}
```

### 模型训练

准备好数据集后，运行 train.py 开始训练，默认的训练配置文件是 <u>configs/train.yaml</u>，默认配置及其含义如下：

```yaml
device: "cuda"        # 设备名称，与 PyTroch 的设备名称保持一致
num-epochs: 200       # 训练迭代次数
num-workers: 8        # DataLoader 数据加载进程数
batch-size: 32        # 批大小

learning-rate: 0.0005        # 学习率
weight-decay: 0.0001         # 权重衰减

use-amp: true                # 是否启动 AMP（自动混合精度），开启有助于减少显存占用并加速训练
use-augment: true            # 是否启用图像数据增强
load-pretrained: true        # 是否使用预训练参数初始化模型权重
load-checkpoint: false       # 是否加载 checkpoint 继续训练，若为 true 则从 load-path 加载模型权重，反之则使用初始化模型权重开始训练

load-path: "checkpoints/last-ckpt1.pt"    # 初始模型加载路径
best-path: "checkpoints/best-ckpt1.pt"    # 当前验证集最优模型保存路径
last-path: "checkpoints/last-ckpt1.pt"    # 最后一次训练模型保存路径
```

也可以基于我训练的模型进行进一步调优，模型权重文件在本项目 Releases 中公布。

### 模型评估

模型训练完成后，运行 eval.py 进行评估。这将会分别计算模型在测试集上的精确率 (Precision)，召回率 (Recall)，F1-score 以及预测结果图和实际分割掩码图的交并比 (IoU) 检测精度指标。默认的配置文件为 <u>configs/eval.yaml</u>，默认配置及其含义如下：

```yaml
device: "cuda"                                     # 设备名称，与 PyTroch 的设备名称保持一致
checkpoint-path: "checkpoints/best-ckpt3.pt"       # 待评估模型加载路径
batch-size: 32                                     # 批大小
num-workers: 8                                     # DataLoader 数据加载进程数
use-amp: true                                      # 是否启动 AMP（自动混合精度），开启有助于减少显存占用并加速推理
```

## 后续计划

- 当前的模型结构作为一个 Baseline，后续将会以此为基础重点在检测精度和计算效率的提升方面尝试对模型结构和算法进行改进。

- 完成基于 ONNX Runtime 的 C++ 模型推理部署程序。

<u>*由于需要加紧完成毕业设计，以上后续计划暂缓执行。*</u>


