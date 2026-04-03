import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from timm import create_model
from torch.cuda.amp import autocast, GradScaler  # 引入混合精度加速

# ================= 性能榨干配置区 =================
# 指向你生成的新 4 分类数据集
BASE_DIR = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

BATCH_SIZE = 64  # 保持与主模型相同的 Batch Size
NUM_WORKERS = 4  # 开启 4 个 CPU 进程
EPOCHS = 15  # 统一迭代次数为 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ResNet-50 对比实验引擎启动，使用设备: {DEVICE}")


# ==================================================

def main():
    # 数据增强策略 (严格保持一致以控制变量)
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集并按 8:2 划分
    full_dataset = datasets.ImageFolder(TRAIN_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    #核心修改 1：加载 ResNet-50 模型
    # 开启 pretrained=True 加载预训练权重，保证 15 个 Epoch 内能收敛到较好的基准水平
    print("⏳ 正在构建 ResNet-50 模型...")
    model = create_model("resnet50", pretrained=True, num_classes=4)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0

    print(f"开始训练 ResNet-50，共计 {EPOCHS} 个 Epoch...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / total
        train_acc = correct / total
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"\n Epoch {epoch + 1} 总结:")
        print(f"   训练集 - Loss: {avg_train_loss:.4f}, 准确率: {train_acc * 100:.2f}%")
        print(f"   验证集 - Loss: {avg_val_loss:.4f}, 准确率: {val_acc * 100:.2f}%")

        #  核心修改 2：独立保存权重，避免覆盖主模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet50_best.pth")
            print("  发现最佳模型！权重已保存为 resnet50_best.pth")

    print(f"\n ResNet-50 对比实验训练结束！")
    print(f"最高验证集准确率: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()