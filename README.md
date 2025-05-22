## 项目简介

这是一个基于PyTorch的手写数字识别系统，使用MNIST数据集进行训练。该项目包含了从模型训练到部署的完整流程，可以识别手写的0-9数字。
使用 Github Copilot 进行协作开发。

## 功能特点

- 使用DenseNet神经网络模型进行手写数字识别
- 支持GPU加速训练(CUDA)和CPU训练
- 包含模型训练、转换和预测功能
- 提供图形界面(GUI)供用户手动绘制数字进行识别
- 支持导出为独立可执行程序

## 项目结构

```
├── main.py               # 训练脚本
├── model.py              # 模型定义
├── predict_cpu.py        # CPU版本预测程序
├── predict_cuda.py       # GPU版本预测程序
├── convert_model.py      # 模型格式转换脚本
├── model.pth             # 训练好的PyTorch模型
├── model.onnx            # 转换后的ONNX格式模型
├── requirements.txt      # 依赖库列表
├── training_metrics.png  # 训练过程指标可视化
└── datasets/             # 数据集目录
    └── MNIST/            # MNIST数据集
```

## 环境要求

通过以下命令安装必要的依赖：

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch
- torchvision
- matplotlib
- numpy
- Pillow (PIL)
- tkinter (GUI界面)

## 使用说明

### 1. 训练模型

运行以下命令训练模型：

```bash
python main.py
```

这将训练一个手写数字识别模型，并将模型保存为model.pth，同时生成训练过程指标可视化图表training_metrics.png。

### 2. 转换模型格式

将PyTorch模型转换为ONNX格式：

```bash
python convert_model.py
```

### 3. 运行预测程序

#### CPU版本：

```bash
python predict_cpu.py
```

#### GPU版本（需要CUDA支持）：

```bash
python predict_cuda.py
```

这将启动一个GUI界面，您可以在其中手动绘制数字，程序会实时进行识别。

## 模型说明

该项目使用了DenseNet架构的卷积神经网络来识别手写数字，定义在model.py中。DenseNet具有以下特点和优势：

### DenseNet架构特点
- **密集连接**: 每个层直接与之前所有层相连，实现了特征的高效重用
- **减轻梯度消失**: 通过密集连接提供了更直接的梯度流，有助于深层网络的训练
- **参数效率**: 同等性能下比传统CNN使用更少的参数，减少了过拟合风险
- **特征融合**: 有效结合低层和高层特征，提高特征表达能力

### 模型结构
- **初始卷积层**: 7×7卷积+BN+ReLU+最大池化，减小特征图尺寸并增加通道数
- **密集块(DenseBlock)**: 包含多个卷积块，每个卷积块的输出与之前所有层的输出连接
- **过渡层(Transition Block)**: 连接两个密集块，通过1×1卷积和平均池化降低特征图分辨率和通道数
- **最终分类层**: BN+ReLU+自适应平均池化+全连接层，输出10个类别概率

通过这种架构设计，模型能够有效学习手写数字的特征表示，提高识别准确率。

## 打包为可执行程序

项目已使用PyInstaller打包为独立的可执行程序，位于predict_cpu目录下。

## 训练结果

训练过程中的损失和准确率变化可在training_metrics.png中查看，包括：
- 训练损失
- 训练准确率
- 测试准确率
