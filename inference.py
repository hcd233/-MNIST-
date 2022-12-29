import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
from PIL import Image, ImageDraw
from train import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("./model_parameter.pkl"))
model.to(device)


# 从画布获取输入并压缩为 28x28 像素的图像
def get_inputs_from_canvas():
    image = Image.new("RGB", (784, 784), "black")
    draw = ImageDraw.Draw(image)
    for item in canvas.find_all():
        coords = canvas.coords(item)
        draw.rectangle(coords, fill="white")
    image = image.resize((28, 28), resample=Image.BICUBIC)
    image = image.convert("L")
    inputs = list(image.getdata())
    inputs = [x / 8 for x in inputs]
    for i in range(len(inputs)):
        if inputs[i] == 0.:
            inputs[i] = -0.4242
    return inputs


# 定义推理函数
def infer():
    # 从画布获取输入
    inputs = get_inputs_from_canvas()
    # 进行推理
    inputs = torch.tensor(inputs).unsqueeze(0).unsqueeze(0).to(device)
    output = model(inputs.reshape((-1, 1, 28, 28)))
    print(inputs)
    _, predicted = torch.max(output, 1)
    # 更新文本标签
    print("Confidence:{:.6f}".format(float(100. * torch.exp(_) / torch.sum(torch.exp(output)))))
    print("Prediction:", predicted.item())
    label_text.set("Prediction: {}".format(predicted.item()))
    # 更新画布
    root.update()


# 定义鼠标按下事件的回调函数
def on_mouse_down(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y


# 定义鼠标移动事件的回调函数
def on_mouse_move(event):
    global start_x, start_y
    canvas.create_line(start_x, start_y, event.x, event.y, width=13)
    start_x = event.x
    start_y = event.y


# 定义清除按钮的回调函数
def on_clear_button_click():
    canvas.delete("all")


if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    # 创建画布
    canvas = tk.Canvas(root, width=784, height=784, bg='white')
    canvas.pack()
    label_text = tk.StringVar()
    label = tk.Label(root, textvariable=label_text)
    label.pack()
    # 全局变量，用于记录鼠标的坐标
    start_x = None
    start_y = None
    # 监听鼠标事件
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    # 创建清除按钮
    clear_button = tk.Button(root, text="Clear", command=on_clear_button_click)
    clear_button.pack()
    # 创建按钮
    button = tk.Button(root, text="Infer", command=infer)
    button.pack()
    # 启动消息循环
    root.mainloop()
