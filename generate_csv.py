import os
import pandas as pd

# 配置数据集路径
DATASET_DIR = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset\train"
OUTPUT_CSV = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset\driver_imgs_list.csv"

data = []
# 扫描 c0 到 c6 里的所有图片
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({"classname": class_name, "img": img_name})

# 生成 CSV 文件
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"成功生成全新标签文件，共找到 {len(df)} 张图片，保存至: {OUTPUT_CSV}")