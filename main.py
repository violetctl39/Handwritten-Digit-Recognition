# 导入PyTorch核心库
import torch
from torchvision import datasets, transforms  # 用于加载和转换图像数据集
from torch.utils.data import DataLoader       # 数据加载器，用于批量加载数据
from torchvision.datasets import MNIST        # MNIST手写数字数据集

# 导入PyTorch神经网络相关模块
from torch import nn                          # 神经网络库
from torch import optim                       # 优化器库
from model import MyModel                     # 导入自定义模型结构

# 导入绘图库
import matplotlib.pyplot as plt

# 训练参数设置
epoch = 10       # 训练轮数
lr = 0.01        # 学习率
batch_size = 64 # 批量大小

# 加载MNIST训练数据，将图像转换为张量，像素值归一化到[0,1]
train_data = MNIST(root='./datasets', train=True, transform=transforms.ToTensor())
# 创建训练数据加载器
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# 加载MNIST测试数据
test_data = MNIST(root="./datasets", train=False, transform=transforms.ToTensor())
# 创建测试数据加载器
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 实例化模型
net = MyModel()
# 检测是否有GPU可用，如果有则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型移动到指定设备(CPU或GPU)
net.to(device)
# 创建Adam优化器，学习率为0.01
optimizer = optim.Adam(net.parameters(), lr=lr)
# 使用交叉熵损失函数，适用于分类任务
loss = nn.CrossEntropyLoss()

def train(net, train_loader, test_loader, optimizer, loss, device, epoch):
    """
    训练神经网络模型的函数
    
    参数:
    net - 神经网络模型
    train_loader - 训练数据加载器
    test_loader - 测试数据加载器
    optimizer - 优化器
    loss - 损失函数
    device - 计算设备(CPU或GPU)
    epoch - 训练轮数
    
    返回:
    train_loss - 每轮的训练损失列表
    train_acc - 每轮的训练准确率列表
    test_acc - 每轮的测试准确率列表
    """
    train_loss = []  # 存储每轮的训练损失
    train_acc = []   # 存储每轮的训练准确率
    test_acc = []    # 存储每轮的测试准确率
    
    # 训练指定轮数
    for e in range(epoch):
        net.train()  # 设置模型为训练模式，启用dropout和批量归一化
        running_loss = 0  # 当前轮的累计损失
        correct = 0       # 当前轮正确预测的样本数
        total = 0         # 当前轮总样本数
        
        # 遍历训练数据集
        for images, labels in train_loader:
            # 将数据移动到指定设备(CPU或GPU)
            images, labels = images.to(device), labels.to(device)
            
            # 清除之前计算的梯度
            optimizer.zero_grad()
            
            # 前向传播：计算预测值
            outputs = net(images)
            
            # 计算损失
            l = loss(outputs, labels)
            
            # 反向传播：计算梯度
            l.backward()
            
            # 更新模型参数
            optimizer.step()
            
            # 累加损失
            running_loss += l.item()
            
            # 获取预测的类别（最大概率对应的索引）
            _, predicted = torch.max(outputs.data, 1)
            
            # 累加样本总数
            total += labels.size(0)
            
            # 累加预测正确的样本数
            correct += (predicted == labels).sum().item()
        
        # 计算当前轮的平均损失并存储
        train_loss.append(running_loss / len(train_loader))
        
        # 计算当前轮的训练准确率并存储
        train_acc.append(correct / total)
        
        # 在测试集上评估模型，存储测试准确率
        test_acc.append(test(net, test_loader, loss, device))
        
        # 打印当前轮的训练信息
        print(f"Epoch {e+1}/{epoch}, Loss: {train_loss[-1]}, Accuracy: {train_acc[-1]}, Test Accuracy: {test_acc[-1]}")
    
    print("Training complete.")
    return train_loss, train_acc, test_acc

def test(net, test_loader, loss, device):
    """
    在测试集上评估模型性能的函数
    
    参数:
    net - 神经网络模型
    test_loader - 测试数据加载器
    loss - 损失函数
    device - 计算设备(CPU或GPU)
    
    返回:
    test_acc - 测试集上的准确率
    """
    net.eval()  # 设置模型为评估模式，禁用dropout和批量归一化
    correct = 0  # 预测正确的样本数
    total = 0    # 总样本数
    
    # 禁用梯度计算，提高推理速度和减少内存使用
    with torch.no_grad():
        # 遍历测试数据集
        for images, labels in test_loader:
            # 将数据移动到指定设备(CPU或GPU)
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播：计算预测值
            outputs = net(images)
            
            # 计算损失(这里没有使用损失值)
            l = loss(outputs, labels)
            
            # 获取预测的类别（最大概率对应的索引）
            _, predicted = torch.max(outputs.data, 1)
            
            # 累加样本总数
            total += labels.size(0)
            
            # 累加预测正确的样本数
            correct += (predicted == labels).sum().item()
    
    # 计算测试准确率
    test_acc = correct / total
    return test_acc

# 导入os模块用于文件操作
import os

# 主程序入口
if __name__ == "__main__":
    # 训练模型，获取训练损失、训练准确率和测试准确率
    train_loss, train_acc, test_acc = train(net, train_loader, test_loader, optimizer, loss, device, epoch)
    
    # 创建model目录(如果不存在)
    os.makedirs("model", exist_ok=True)
    
    # 保存训练好的模型参数
    torch.save(net.state_dict(), "model\\model.pth")

    # 创建一个大小为10x6的图形
    plt.figure(figsize=(10, 6))
    
    # 创建x轴数据：轮数范围
    epochs = range(1, epoch + 1)
    
    # 绘制训练损失曲线(蓝色)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    
    # 绘制训练准确率曲线(红色)
    plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
    
    # 绘制测试准确率曲线(绿色)
    plt.plot(epochs, test_acc, 'g-', label='Test Accuracy')
    
    # 添加x轴标签
    plt.xlabel('Epoch')
    
    # 添加y轴标签
    plt.ylabel('Value')
    
    # 添加图表标题
    plt.title('Training and Evaluation Metrics')
    
    # 添加图例
    plt.legend()
    
    # 显示网格线
    plt.grid(True)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像为PNG，DPI为300
    plt.savefig('training_metrics.png', dpi=300)
