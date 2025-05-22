import torch
from torch import nn

class conv_block(nn.Module):
    """
    卷积块 - DenseNet的基本组件
    包含BN-ReLU-Conv结构，实现特征提取
    
    参数:
    in_channels - 输入通道数
    num_channels - 输出通道数
    """
    def __init__(self, in_channels, num_channels):
        super(conv_block, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),  # 批量归一化，稳定训练过程
            nn.ReLU(),                    # 激活函数，引入非线性
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),  # 3x3卷积，padding=1保持特征图大小
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

class DenseBlock(nn.Module):
    """
    密集块 - DenseNet的核心结构
    由多个卷积块组成，每个卷积块的输出都与之前所有层的输出连接
    
    参数:
    num_convs - 卷积块数量
    in_channels - 输入通道数
    num_channels - 每个卷积层增加的通道数(增长率)
    """
    def __init__(self, num_convs, in_channels, num_channels):
        super(DenseBlock, self).__init__()
        layers = []
        # 创建num_convs个卷积块
        for i in range(num_convs):
            # 每个卷积块的输入通道数递增，体现密集连接特性
            layers.append(conv_block(in_channels + i * num_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 实现密集连接：每层的输出与之前所有层的输出拼接
        for layer in self.net:
            out = layer(x)           # 通过当前层
            x = torch.cat((x, out), dim=1)  # 沿通道维度拼接特征
        return x  # 返回密集连接后的特征

class Transition_block(nn.Module):
    """
    过渡块 - 连接两个密集块，降低特征图分辨率并减少通道数
    包含BN-ReLU-1x1Conv-AvgPool结构
    
    参数:
    in_channels - 输入通道数
    out_channels - 输出通道数，通常为输入通道数的一半
    """
    def __init__(self,in_channels, out_channels):
        super(Transition_block, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),  # 批量归一化
            nn.ReLU(),                    # 激活函数
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1x1卷积减少通道数
            nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2平均池化减小特征图尺寸
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

class MyModel(nn.Module):
    """
    基于DenseNet架构的手写数字识别模型
    由初始卷积层、多个密集块和过渡块、以及最终分类层组成
    
    网络结构:
    1. 初始卷积层(Conv-BN-ReLU-MaxPool)
    2. 多个密集块和过渡块交替
    3. 最终分类层(BN-ReLU-AdaptiveAvgPool-Flatten-Linear)
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # 初始卷积层，将1通道图像转为64通道特征图，并降低分辨率
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 7x7卷积，步长2
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),           # 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3x3最大池化，步长2
        )
        
        # 设置通道数和增长率参数
        num_channels, growth_rate = 64, 32
        # 定义每个密集块中的卷积块数量
        num_convs_in_dense_blocks = [4, 4]
        # 创建模块列表，存储密集块和过渡块
        self.blks = nn.ModuleList()
        
        # 构建密集块和过渡块
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            # 添加密集块
            self.blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 更新通道数：原通道数 + 增长率*卷积块数量
            num_channels += num_convs * growth_rate
            
            # 除了最后一个密集块外，每个密集块后添加过渡块
            if i != len(num_convs_in_dense_blocks) - 1:
                # 添加过渡块，将通道数减半
                self.blks.append(Transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        
        # 最终分类层
        self.fb = nn.Sequential(
            nn.BatchNorm2d(num_channels),  # 批量归一化
            nn.ReLU(),                    # 激活函数
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，输出1x1特征图
            nn.Flatten(),                 # 展平特征
            nn.Linear(num_channels, 10)   # 全连接层，输出10个类别(0-9数字)
        )
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x - 输入图像张量，形状为[batch_size, 1, 28, 28]
        
        返回:
        预测结果，形状为[batch_size, 10]
        """
        # 通过初始卷积层
        x = self.b1(x)
        # 依次通过所有密集块和过渡块
        for blk in self.blks:
            x = blk(x)
        # 通过最终分类层
        x = self.fb(x)
        return x

