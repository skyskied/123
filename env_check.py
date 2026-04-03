# env_check.py
import importlib
import subprocess
import sys

# 所需依赖列表（根据需求扩展）
REQUIRED_PACKAGES = [
    "cv2", "dlib", "matplotlib", "mmselfsup", "pytorchvideo", "timm",
    "onnx", "onnxruntime", "pyqt5", "pyttsx3", "pandas", "flask", "torch", "torchvision"
]

def check_and_install_packages():
    for pkg in REQUIRED_PACKAGES:
        try:
            # 检查包是否安装（处理别名，如cv2对应opencv-python）
            if pkg == "cv2":
                importlib.import_module("cv2")
            else:
                importlib.import_module(pkg)
            print(f"✅ {pkg} 已安装")
        except ImportError:
            print(f"❌ 缺失 {pkg}，正在安装...")
            # 处理特殊包的安装命令
            install_cmd = f"pip install {pkg}"
            if pkg == "cv2":
                install_cmd = "pip install opencv-python"
            elif pkg == "mmselfsup":
                install_cmd = "git clone https://github.com/open-mmlab/mmselfsup.git && cd mmselfsup && pip install -v -e ."
            # 执行安装命令
            subprocess.run(install_cmd, shell=True, check=True)
    print("✅ 所有依赖已安装完成")

if __name__ == "__main__":
    check_and_install_packages()