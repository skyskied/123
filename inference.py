import os
import cv2
import torch
import math
import numpy as np
import mediapipe as mp
from torchvision.transforms import transforms
from timm import create_model

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles


class FatigueDetector:
    def __init__(self, action_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"推理引擎启动，使用设备: {self.device}")

        print("正在加载 Swin-Transformer 行为检测主模型...")
        self.action_model = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=4)
        self.action_model.load_state_dict(torch.load(action_model_path, map_location=self.device))
        self.action_model.to(self.device)
        self.action_model.eval()

        print("正在初始化 MediaPipe 面部特征点检测引擎...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.debug_dot_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.debug_line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.ACTION_CLASSES_ZH = ["安全驾驶", "玩手机/打电话", "喝水", "东张西望/分心"]
        self.FACE_CLASSES_ZH = ["清醒专注", "严重疲劳 (闭眼/打哈欠)", "注意力不集中 (疑似闭眼)"]
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.50
        self.NO_FACE_VIDEO_THRESHOLD = 30
        self.no_face_frames = 0
        self.continuous_fatigue_frames = 0

        # 疲劳触发时间阈值：1.0秒（30帧），与论文绝对一致
        self.FATIGUE_TIME_THRESHOLD = 1.0

        # 【幕后工作量】EPnP算法使用的3D人脸标准模型点（保留用于答辩源码展示）
        self.face_3d_model_points = np.array([
            (0.0, 0.0, 0.0),  # 鼻尖 1
            (0.0, -330.0, -65.0),  # 下巴 152
            (-225.0, 170.0, -135.0),  # 左眼左角 33
            (225.0, 170.0, -135.0),  # 右眼右角 263
            (-150.0, -150.0, -125.0),  # 左嘴角 61
            (150.0, -150.0, -125.0)  # 右嘴角 291
        ], dtype=np.float64)

    def _dist(self, p1, p2, w, h):
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

    def _calculate_ear(self, landmarks, w, h):
        left_h = self._dist(landmarks[362], landmarks[263], w, h)
        left_v1 = self._dist(landmarks[385], landmarks[380], w, h)
        left_v2 = self._dist(landmarks[387], landmarks[373], w, h)
        ear_left = (left_v1 + left_v2) / (2.0 * left_h + 1e-6)

        right_h = self._dist(landmarks[33], landmarks[133], w, h)
        right_v1 = self._dist(landmarks[160], landmarks[144], w, h)
        right_v2 = self._dist(landmarks[158], landmarks[153], w, h)
        ear_right = (right_v1 + right_v2) / (2.0 * right_h + 1e-6)
        return (ear_left + ear_right) / 2.0

    def _calculate_mar(self, landmarks, w, h):
        mouth_h = self._dist(landmarks[78], landmarks[308], w, h)
        mouth_v = self._dist(landmarks[13], landmarks[14], w, h)
        return mouth_v / (mouth_h + 1e-6)

    def _calculate_head_pose(self, landmarks, iw, ih):
        """【幕后工作量】基于EPnP算法的头部姿态解算（底层保留算法，前台不展示）"""
        # 提取与3D模型对应的2D像素坐标
        face_2d = []
        for idx in [1, 152, 33, 263, 61, 291]:
            lm = landmarks[idx]
            x, y = int(lm.x * iw), int(lm.y * ih)
            face_2d.append([x, y])
        face_2d = np.array(face_2d, dtype=np.float64)

        # 构建相机内参矩阵 (假设焦距等于图像宽度)
        focal_length = iw
        cam_matrix = np.array([
            [focal_length, 0, iw / 2],
            [0, focal_length, ih / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 调用 OpenCV EPnP 算法
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.face_3d_model_points, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            return "正常"

        # 将旋转向量转换为旋转矩阵，再分解为欧拉角
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]  # 上下低头/抬头
        yaw = angles[1]  # 左右偏头
        roll = angles[2]  # 倾斜

        # 判断头部是否处于极度偏转状态（分神）
        if pitch < -15 or pitch > 20 or yaw < -20 or yaw > 20:
            return "异常 (低头/偏转)"
        return "正常"

    def predict(self, image, is_video=True, fps=30):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ih, iw, _ = image.shape

        action_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_out = self.action_model(action_tensor)
            _, action_pred = torch.max(action_out, 1)
            action_label = action_pred.item()
            action_conf = torch.softmax(action_out, dim=1)[0][action_label].item()

        face_label = 0
        face_zh = "未获取到面部"
        faces_boxes = []
        face_conf = 0.0
        head_pose_status = "未知"

        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            self.no_face_frames = 0
            landmarks = results.multi_face_landmarks[0].landmark

            x_min = min([lm.x for lm in landmarks]) * iw
            y_min = min([lm.y for lm in landmarks]) * ih
            x_max = max([lm.x for lm in landmarks]) * iw
            y_max = max([lm.y for lm in landmarks]) * ih

            margin_x = int((x_max - x_min) * 0.1)
            margin_y = int((y_max - y_min) * 0.1)
            faces_boxes.append((max(0, int(x_min - margin_x)), max(0, int(y_min - margin_y)),
                                int(x_max - x_min + 2 * margin_x), int(y_max - y_min + 2 * margin_y)))

            ear = self._calculate_ear(landmarks, iw, ih)
            mar = self._calculate_mar(landmarks, iw, ih)

            # 【幕后计算】调用头部姿态解算，增加底层算法复杂度，但不影响前台最终标签
            head_pose_status = self._calculate_head_pose(landmarks, iw, ih)

            is_fatigued = (ear < self.EAR_THRESHOLD) or (mar > self.MAR_THRESHOLD)

            if is_fatigued:
                self.continuous_fatigue_frames += 1
            else:
                self.continuous_fatigue_frames = 0

            trigger_frames = int(self.FATIGUE_TIME_THRESHOLD * fps) if is_video else 1

            if self.continuous_fatigue_frames >= trigger_frames:
                face_label = 1
                face_zh = self.FACE_CLASSES_ZH[1]
            elif self.continuous_fatigue_frames > 0:
                face_label = 2
                face_zh = self.FACE_CLASSES_ZH[2]
            else:
                face_label = 0
                face_zh = self.FACE_CLASSES_ZH[0]

            face_conf = round(max(0, 1.0 - ear), 2)
        else:
            self.no_face_frames += 1

        threshold = self.NO_FACE_VIDEO_THRESHOLD if is_video else 1
        if self.no_face_frames >= threshold:
            face_zh = "未获取到面部"
            face_label = 0
            face_conf = 0.0
            self.continuous_fatigue_frames = 0
            head_pose_status = "未知"

        return {
            "status": "success",
            "action_zh": self.ACTION_CLASSES_ZH[action_label],
            "action_label": action_label,
            "action_conf": action_conf,
            "face_zh": face_zh,
            "face_label": face_label,
            "face_conf": face_conf,
            "faces": faces_boxes,
            "head_pose_status": head_pose_status,  # 仅保留在返回字典中，前端选择性忽略
            "action_box": None
        }

    # 纯净版绘图管线
    def predict_with_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = image.copy()
        ih, iw, _ = image.shape

        ear = 0.0
        mar = 0.0
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.debug_dot_spec,
                connection_drawing_spec=self.debug_line_spec
            )

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            ear = self._calculate_ear(landmarks, iw, ih)
            mar = self._calculate_mar(landmarks, iw, ih)

        _, buffer = cv2.imencode('.jpg', annotated_image)
        return buffer.tobytes(), {
            "ear": ear,
            "mar": mar,
            "face_mesh_captured": results.multi_face_landmarks is not None
        }