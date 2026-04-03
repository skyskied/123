import os
import cv2
import shutil
import pandas as pd


# ================= 路径配置 =================
# 你所有图片目前所在的源文件夹
SRC_BASE = r"C:\Users\Administrator\Desktop\bi-she\all"

# 新建专用于训练“纯面部模型”的大本营
DEST_BASE = r"C:\Users\Administrator\Desktop\bi-she\Face_Dataset"
TRAIN_DIR = os.path.join(DEST_BASE, "train")

AWAKE_DIR = os.path.join(TRAIN_DIR, "0_awake")  # 存放清醒的脸
FATIGUE_DIR = os.path.join(TRAIN_DIR, "1_fatigue")  # 存放疲劳的脸

# 确保目标文件夹存在
os.makedirs(AWAKE_DIR, exist_ok=True)
os.makedirs(FATIGUE_DIR, exist_ok=True)
# ============================================

# 初始化 OpenCV 人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_and_save(src_folder, dest_folder, prefix):
    """提取人脸并保存"""
    folder_path = os.path.join(SRC_BASE, src_folder)
    if not os.path.exists(folder_path):
        print(f" 找不到文件夹: {folder_path}，已跳过。")
        return

    files = os.listdir(folder_path)
    print(f"\n 正在处理 {src_folder} (共 {len(files)} 张图片)...")

    saved = 0
    for img_name in files:
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(folder_path, img_name)

        import numpy as np
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 如果检测到人脸
        if len(faces) > 0:
            # 取面积最大的一张脸
            (x, y, w, h) = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]

            #  核心：稍微扩大裁剪框，包含一点额头和下巴（增加 20% 边缘）
            margin_x, margin_y = int(w * 0.2), int(h * 0.2)
            x_min = max(0, x - margin_x)
            y_min = max(0, y - margin_y)
            x_max = min(img.shape[1], x + w + margin_x)
            y_max = min(img.shape[0], y + h + margin_y)

            # 裁剪并保存
            face_img = img[y_min:y_max, x_min:x_max]

            # 忽略太小的无效残影 (低于 50x50 像素)
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue

            save_name = f"{prefix}_{img_name}"
            save_path = os.path.join(dest_folder, save_name)

            cv2.imencode('.jpg', face_img)[1].tofile(save_path)
            saved += 1

    print(f" {src_folder} 处理完毕！成功提取 {saved} 张纯净人脸。")

def copy_direct(src_folder, dest_folder, prefix):
    """对于 CEW 这种已经是纯脸的图片，直接复制"""
    folder_path = os.path.join(SRC_BASE, src_folder)
    if not os.path.exists(folder_path): return

    files = os.listdir(folder_path)
    print(f"\n 正在直接复制 {src_folder} (共 {len(files)} 张图片)...")

    saved = 0
    for img_name in files:
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            src_path = os.path.join(folder_path, img_name)
            save_path = os.path.join(dest_folder, f"{prefix}_{img_name}")
            shutil.copy(src_path, save_path)
            saved += 1
    print(f" {src_folder} 复制完毕！")

def generate_csv():
    """生成用于训练的 CSV 标签表"""
    print("\n 正在生成 Face_Dataset 的 driver_imgs_list.csv ...")
    data = []

    for class_name in ["0_awake", "1_fatigue"]:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({"classname": class_name, "img": img_name})

    csv_path = os.path.join(DEST_BASE, "driver_imgs_list.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f" 全部搞定！共记录 {len(df)} 张人脸样本，已保存至: {csv_path}")


if __name__ == "__main__":
    print(" 开始构建终极人脸二分类数据集...")

    # 1. 裁剪 RLDD 视频抽出的半身照
    crop_and_save("c7", AWAKE_DIR, "rldd_awake")
    crop_and_save("c8", FATIGUE_DIR, "rldd_fatigue")

    # 2. 直接复制 CEW 的纯大头照
    copy_direct("c11", AWAKE_DIR, "cew_awake")
    copy_direct("c12", FATIGUE_DIR, "cew_fatigue")

    # 3. 生成标签
    generate_csv()
