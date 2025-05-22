import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import onnxruntime as ort
import os
import sys

def resource_path(relative_path):
    """ 获取资源绝对路径 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

ort_session = ort.InferenceSession(resource_path('model.onnx'))

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
        self.button_clear = tk.Button(self, text="清空", command=self.clear)  # 创建清空按钮，点击时调用clear方法
        self.button_clear.pack()  # 将清空按钮添加到界面
        self.image1 = Image.new("L", (400, 400), 'white')  # 创建一个400x400的灰度图像，背景为白色
        self.draw = ImageDraw.Draw(self.image1)  # 创建绘图对象，用于在图像上绘制
        self.canvas.bind("<B1-Motion>", self.paint)  # 绑定鼠标左键拖动事件到paint方法

    def paint(self, event):
        """处理鼠标绘图事件，在画布和图像上同时绘制"""
        x1, y1 = (event.x - 8), (event.y - 8)  # 计算椭圆左上角坐标
        x2, y2 = (event.x + 8), (event.y + 8)  # 计算椭圆右下角坐标
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=8)  # 在画布上绘制黑色椭圆
        self.draw.ellipse([x1, y1, x2, y2], fill='black')  # 在图像上同时绘制相同的椭圆

    def clear(self):
        """清空画布和图像内容"""
        self.canvas.delete("all")  # 删除画布上的所有内容
        self.draw.rectangle([0, 0, 400, 400], fill='white')  # 用白色矩形覆盖整个图像
        self.label_result.config(text="结果：")  # 重置结果标签文本

    def predict(self):
        """处理图像并使用ONNX模型进行预测"""
        img = self.image1.resize((28, 28))  # 将图像调整为28x28像素(MNIST标准尺寸)
        img = ImageOps.invert(img)  # 反转图像颜色(MNIST数据集是黑底白字)
        img = np.array(img, dtype=np.float32) / 255.0  # 转换为numpy数组并归一化到0-1
        img = img.reshape(1, 1, 28, 28)  # 调整维度为模型输入格式(批次,通道,高度,宽度)
        ort_inputs = {ort_session.get_inputs()[0].name: img}  # 创建ONNX模型输入字典
        ort_outs = ort_session.run(None, ort_inputs)  # 运行模型推理获取输出
        pred = np.argmax(ort_outs[0])
        self.label_result.config(text=f"结果：{pred}")

if __name__ == "__main__":
    app = App()
    app.mainloop()