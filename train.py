import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from timm import create_model
from torch.cuda.amp import autocast, GradScaler #  引入混合精度加速

# ================= 性能榨干配置区 =================
# 指向你刚刚生成的新 4 分类数据集
BASE_DIR = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

BATCH_SIZE = 64        # 将显卡占用拉满
NUM_WORKERS = 4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" 深度加速引擎启动，使用设备: {DEVICE}")
# ==================================================

def main():
    # 数据增强
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

    #  核心优化 1：使用官方底层的 ImageFolder，比之前读取 CSV 快几倍！
    print(f"⏳ 正在扫描数据集: {TRAIN_DIR}")
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    print(f" 数据集加载成功，共找到 {len(full_dataset)} 张图片，分类为: {full_dataset.classes}")

    # 80%训练集,20%验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 替换验证集的 transform
    val_dataset.dataset.transform = val_transform

    #  核心优化 2：开启 num_workers 和 pin_memory 加速数据流
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    #  核心修改：针对你新洗出的 4 分类专属模型
    print("⏳ 正在构建 Swin Transformer 模型并加载官方预训练权重...")
    model = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=4)
    model = model.to(DEVICE)
    print(" 模型构建完毕")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # 提升了一点点学习率以适应大Batch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    #  核心优化 3：初始化混合精度梯度缩放器
    scaler = GradScaler()
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            
            #  核心优化 4：开启半精度自动转换 (AMP)，计算速度飙升！
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # 使用 scaler 缩放梯度并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                # 验证时同样开启半精度加速
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
        print(f"   验证集 - Loss: {avg_val_loss:.4f}, 准确率: {val_acc * 100:.2f}%\n")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "action_detection_best.pth")
            print(f" 发现新高！动作模型已保存至 action_detection_best.pth (验证准确率: {val_acc * 100:.2f}%)")

    print(f"\n 训练彻底完成！最佳验证准确率: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()