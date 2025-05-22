import torch
from model import MyModel

# 加载模型
model = MyModel()
model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
model.eval()

# 转换为ONNX格式
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, 'model.onnx', 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("模型已转换为ONNX格式")