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
        self.FATIGUE_TIME_THRESHOLD = 2.0

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

        return {
            "status": "success",
            "action_zh": self.ACTION_CLASSES_ZH[action_label],
            "action_label": action_label,
            "action_conf": action_conf,
            "face_zh": face_zh,
            "face_label": face_label,
            "face_conf": face_conf,
            "faces": faces_boxes,
            "action_box": None
        }

    # 纯净版绘图管线：没有文字，没有动作识别
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