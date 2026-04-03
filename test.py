import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model

# 1. 配置路径 (根据你的实际路径调整)
DATA_DIR = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset\train"
MODEL_PATH = "action_detection_best.pth"

# 2. 数据预处理 (必须与验证集保持完全一致)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行准确率评估...")

    # 加载数据集
    dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    # 使用较大的 batch_size 加快评估速度
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 加载模型


    print("⏳ 正在加载 Swin-Tiny 模型权重...")
    model = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # 开启评估模式

    correct = 0
    total = 0

    print("正在计算准确率，请稍候...")
    with torch.no_grad(): # 评估时不计算梯度，节省显存
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print("\n" + "="*30)
    print(f"模型评估完成！")
    print(f"图片总数: {total} 张")
    print(f"预测正确: {correct} 张")
    print(f"最终准确率: {accuracy:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()