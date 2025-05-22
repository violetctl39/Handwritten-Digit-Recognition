import tkinter as tk  # 导入GUI界面库
from PIL import Image, ImageDraw, ImageOps  # 导入图像处理库
import numpy as np  # 导入数值计算库
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块

import os  # 导入操作系统接口模块，用于处理文件路径
import sys  # 导入系统模块，用于获取系统相关信息

from model import MyModel  # 导入自定义的模型类

def resource_path(relative_path):
    """ 获取资源绝对路径 
    将相对路径转换为绝对路径，支持PyInstaller打包后的路径解析
    """
    try:
        # PyInstaller创建临时文件夹并将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        # 如果不是通过PyInstaller运行，使用当前目录作为基础路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 设置计算设备，优先使用GPU(CUDA)，如果不可用则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
model = MyModel().to(device)  # 创建模型实例并移至相应设备(GPU/CPU)
model.load_state_dict(torch.load(resource_path('model.pth'), map_location=device))  # 加载保存的模型参数
model.eval()  # 设置模型为评估模式，禁用掉一些训练时特有的操作（如Dropout）

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.canvas = tk.Canvas(self, width=400, height=400, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="预测", command=self.predict)
        self.button_predict.pack()
        self.label_result = tk.Label(self, text="结果：")
        self.label_result.pack()
        self.button_clear = tk.Button(self, text="清空", command=self.clear)
        self.button_clear.pack()
        self.image1 = Image.new("L", (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        """ 处理绘图事件 """
        x1, y1 = (event.x - 8), (event.y - 8)  # 计算椭圆的左上角坐标
        x2, y2 = (event.x + 8), (event.y + 8)  # 计算椭圆的右下角坐标
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=8)  # 在画布上绘制椭圆
        self.draw.ellipse([x1, y1, x2, y2], fill='black')  # 在图像上绘制椭圆

    def clear(self):
        """ 清空画布和图像 """
        self.canvas.delete("all")  # 清空画布上的所有内容
        self.draw.rectangle([0, 0, 400, 400], fill='white')  # 在图像上绘制白色矩形以清空图像
        self.label_result.config(text="结果：")  # 重置结果标签

    def predict(self):
        """ 处理预测操作 """
        img = self.image1.resize((28, 28))  # 将图像调整为28x28大小
        img = ImageOps.invert(img)  # 反转图像颜色
        img = np.array(img) / 255.0  # 将图像转换为数组并归一化到[0,1]范围
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 转换为张量并调整形状
        with torch.no_grad():  # 在推理时禁用梯度计算
            output = model(img)  # 执行前向传播
            pred = output.argmax(dim=1).item()  # 获取预测结果
        self.label_result.config(text=f"结果：{pred}")  # 更新结果标签

if __name__ == "__main__":
    app = App()
    app.mainloop()