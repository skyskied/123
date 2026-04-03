import os
import shutil

# 你的 Roboflow 数据集绝对路径
SOURCE_IMG_DIR = r"C:\Users\Administrator\Desktop\bi-she\shu_ju_ji\driving.v5i.yolov8\train\images"
SOURCE_LBL_DIR = r"C:\Users\Administrator\Desktop\bi-she\shu_ju_ji\driving.v5i.yolov8\train\labels"

# 新建一个专属的动作分类数据集大本营
TARGET_DIR = r"C:\Users\Administrator\Desktop\bi-she\Action_Classification_Dataset\train"

# 根据 data.yaml 将类别映射为 4 大核心动作文件夹
# 0: Safe, 1: Texting, 2: Phone, 3: Drinking, 4: Reaching Behind, 5: Talking to Passenger
CLASS_MAPPING = {
    0: "0_safe",            # 安全驾驶
    1: "1_phone",           # 玩手机/发短信
    2: "1_phone",           # 打电话 (合并为手机类)
    3: "2_drink",           # 喝水
    4: "3_distracted",      # 向后转身 (合并为注意力不集中)
    5: "3_distracted"       # 与乘客交谈 (合并为注意力不集中)
}

# 创建目标文件夹
for folder in set(CLASS_MAPPING.values()):
    os.makedirs(os.path.join(TARGET_DIR, folder), exist_ok=True)

print("正在洗牌数据集，提取高质量动作分类图片...")
count = 0

for label_file in os.listdir(SOURCE_LBL_DIR):
    if not label_file.endswith('.txt'): continue
    
    with open(os.path.join(SOURCE_LBL_DIR, label_file), 'r') as f:
        lines = f.readlines()
        if not lines: continue
        
        try:
            # 读取 txt 第一行的第一个数字作为类别 ID
            class_id = int(lines[0].split()[0])
            
            # 只提取我们在 MAPPING 里定义的核心动作
            if class_id in CLASS_MAPPING:
                target_folder = CLASS_MAPPING[class_id]
                img_name = label_file.replace('.txt', '.jpg')
                
                src_img_path = os.path.join(SOURCE_IMG_DIR, img_name)
                dst_img_path = os.path.join(TARGET_DIR, target_folder, img_name)
                
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    count += 1
        except Exception as e:
            pass

print(f"数据集洗牌完成！共提取了 {count} 张完美适配论文的分类图片。")
print(f"新数据集路径: {TARGET_DIR}")