import os
import cv2
import numpy as np
import time
import threading
import random
import string
import smtplib
import ssl
import queue
import warnings
import base64
from email.message import EmailMessage

warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from functools import wraps
from PIL import Image, ImageDraw, ImageFont
from inference import FatigueDetector
from database import get_user_by_email, update_password, verify_user, get_all_users, delete_user, update_user_info, \
    get_user_by_id, add_user

current_dir = os.path.dirname(os.path.abspath(__file__))

ACTION_MODEL_PATH = os.path.join(current_dir, "action_detection_best.pth")

if not os.path.exists(ACTION_MODEL_PATH):
    print("错误: 找不到动作模型权重文件 action_detection_best.pth。")
    exit(1)

detector = FatigueDetector(ACTION_MODEL_PATH)

app = Flask(__name__,
            template_folder=os.path.join(current_dir, "templates"),
            static_folder=os.path.join(current_dir, "static"))

app.secret_key = 'fatigue_detection_secret_key_2026'

app.config["UPLOAD_FOLDER"] = os.path.join(current_dir, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

reset_codes = {}
video_buffers = {}
buffer_lock = threading.Lock()


def put_chinese_text(img, text, position, text_color, font_size=30):
    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        try:
            font = ImageFont.truetype("msyh.ttc", font_size, encoding="utf-8")
        except:
            font = ImageFont.load_default()
    draw.text(position, text, fill=text_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_colors_by_label(label):
    if label == 0:
        return (0, 255, 0), (0, 255, 0)
    elif label == 2:
        return (255, 165, 0), (0, 165, 255)
    else:
        return (255, 0, 0), (0, 0, 255)


def background_video_processor(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_fps = 30.0 if original_fps > 30 else original_fps
    skip_ratio = original_fps / target_fps
    extracted_total_frames = int(total_frames / skip_ratio)

    with buffer_lock:
        if filename in video_buffers:
            video_buffers[filename]['total_frames'] = extracted_total_frames
            video_buffers[filename]['fps'] = target_fps

    DETECT_INTERVAL = 3
    extracted_frame_count = 0
    last_result = None
    processing_start_time = time.time()
    processed_frames_in_period = 0

    next_capture_frame = 0.0

    while cap.isOpened():
        if filename not in video_buffers or video_buffers[filename]['stop_event'].is_set():
            break

        seek_target = None
        with buffer_lock:
            if filename in video_buffers and video_buffers[filename].get('seek_target') is not None:
                seek_target = video_buffers[filename]['seek_target']
                video_buffers[filename]['seek_target'] = None
                while not video_buffers[filename]['queue'].empty():
                    try:
                        video_buffers[filename]['queue'].get_nowait()
                    except:
                        pass

        if seek_target is not None:
            original_seek_target = int(seek_target * skip_ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_seek_target)
            next_capture_frame = float(original_seek_target)
            extracted_frame_count = seek_target

        if video_buffers[filename]['queue'].full():
            time.sleep(0.05)
            continue

        current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if original_fps > 30 and current_frame_pos < int(next_capture_frame):
            cap.grab()
            continue

        if original_fps > 30:
            next_capture_frame += skip_ratio

        ret, frame = cap.read()
        if not ret:
            try:
                video_buffers[filename]['queue'].put(None, timeout=1)
            except:
                pass
            break

        processed_frames_in_period += 1
        if time.time() - processing_start_time >= 1.0:
            current_proc_fps = processed_frames_in_period / (time.time() - processing_start_time)
            with buffer_lock:
                if filename in video_buffers:
                    video_buffers[filename]['processing_fps'] = current_proc_fps
            processed_frames_in_period = 0
            processing_start_time = time.time()

        try:
            if extracted_frame_count % DETECT_INTERVAL == 0:
                result = detector.predict(frame, is_video=True, fps=target_fps)
                last_result = result
            else:
                result = last_result if last_result else detector.predict(frame, is_video=True, fps=target_fps)

            if result["status"] == "no_object":
                pass
            else:
                action_text_color, _ = get_colors_by_label(result["action_label"] if result["action_label"] == 0 else 1)
                face_text_color, face_box_color = get_colors_by_label(result["face_label"])

                action_text = f"身体行为: {result['action_zh']} ({result['action_conf']:.2f})"
                face_text = f"面部状态: {result['face_zh']} ({result['face_conf']:.2f})"

                # 剔除了头部姿态的UI渲染，仅保留核心判断
                frame = put_chinese_text(frame, action_text, (10, 20), action_text_color, font_size=32)
                frame = put_chinese_text(frame, face_text, (10, 65), face_text_color, font_size=32)

                for (x, y, w, h) in result["faces"]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), face_box_color, 2)

            fps_text = f"FPS: {target_fps:.0f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            print(f"帧处理出错: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            try:
                video_buffers[filename]['queue'].put(frame_bytes, timeout=1)
                with buffer_lock:
                    if filename in video_buffers:
                        video_buffers[filename]['processed_count'] = extracted_frame_count
            except:
                pass

        extracted_frame_count += 1

    cap.release()


def background_debug_video_processor(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_fps = 30.0 if original_fps > 30 else original_fps
    skip_ratio = original_fps / target_fps
    extracted_total_frames = int(total_frames / skip_ratio)

    with buffer_lock:
        if filename in video_buffers:
            video_buffers[filename]['total_frames'] = extracted_total_frames
            video_buffers[filename]['fps'] = target_fps

    extracted_frame_count = 0
    next_capture_frame = 0.0

    while cap.isOpened():
        if filename not in video_buffers or video_buffers[filename]['stop_event'].is_set():
            break

        if video_buffers[filename]['queue'].qsize() > 50:
            while not video_buffers[filename]['queue'].empty():
                try:
                    video_buffers[filename]['queue'].get_nowait()
                except:
                    pass

        ret, frame = cap.read()
        if not ret:
            try:
                video_buffers[filename]['queue'].put(None, timeout=1)
            except:
                pass
            break

        current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if original_fps > 30 and current_frame_pos < int(next_capture_frame):
            cap.grab()
            continue

        if original_fps > 30:
            next_capture_frame += skip_ratio

        try:
            frame_bytes, _ = detector.predict_with_landmarks(frame)
            video_buffers[filename]['queue'].put(frame_bytes, timeout=1)
            with buffer_lock:
                if filename in video_buffers:
                    video_buffers[filename]['processed_count'] = extracted_frame_count
        except Exception as e:
            print(f"调试帧处理出错: {e}")

        extracted_frame_count += 1

    cap.release()


def video_stream_generator(filename):
    timeout = 10
    start_wait = time.time()
    while filename not in video_buffers:
        if time.time() - start_wait > timeout:
            return
        time.sleep(0.1)

    buffer_data = video_buffers[filename]
    frame_queue = buffer_data['queue']

    target_fps = buffer_data.get('fps', 30)
    PRE_BUFFER_COUNT = int(target_fps * 3)

    while frame_queue.qsize() < PRE_BUFFER_COUNT and frame_queue.qsize() < 100:
        if not buffer_data['thread'].is_alive() and frame_queue.empty():
            break
        time.sleep(0.1)

    while True:
        if filename not in video_buffers or buffer_data['stop_event'].is_set():
            break

        if buffer_data.get('is_paused', False):
            time.sleep(0.1)
            continue

        if frame_queue.empty():
            if buffer_data['thread'].is_alive():
                time.sleep(0.05)
                continue
            else:
                break

        speed = buffer_data.get('speed', 1.0)
        frame_duration = 1.0 / (target_fps * speed)

        start_time = time.time()

        frame_data = frame_queue.get()
        if frame_data is None:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        with buffer_lock:
            buffer_data['current_play_frame'] += 1

        elapsed = time.time() - start_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)


def debug_video_stream_generator(filename):
    timeout = 10
    start_wait = time.time()
    while filename not in video_buffers:
        if time.time() - start_wait > timeout:
            return
        time.sleep(0.1)

    buffer_data = video_buffers[filename]
    frame_queue = buffer_data['queue']
    target_fps = buffer_data.get('fps', 30)

    while True:
        if filename not in video_buffers or buffer_data['stop_event'].is_set():
            break

        if frame_queue.empty():
            if buffer_data['thread'].is_alive():
                time.sleep(0.05)
                continue
            else:
                break

        frame_duration = 1.0 / target_fps
        start_time = time.time()

        frame_data = frame_queue.get()
        if frame_data is None:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        elapsed = time.time() - start_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_user_by_id(session.get('user_id'))
        if not user or not user['is_admin']:
            return jsonify({"success": False, "message": "权限被拒绝：需要系统管理员权限方可访问调试模式。"}), 403
        return f(*args, **kwargs)

    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        success, message, user = verify_user(username, password)

        if success:
            session['username'] = username
            session['is_admin'] = user['is_admin']
            session['user_id'] = user['id']
            session['avatar'] = user.get('avatar')
            return jsonify({"success": True, "message": "登录成功！正在为您跳转...", "redirect_url": url_for('index')})
        else:
            return jsonify({"success": False, "message": message})
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if username and email and password:
            success, message = add_user(username, password, email)
            if success:
                return jsonify({"success": True, "message": message, "redirect_url": url_for('login')})
            else:
                return jsonify({"success": False, "message": message})
        else:
            return jsonify({"success": False, "message": "请填写完整的注册信息！"})
    return render_template('register.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if email:
            user = get_user_by_email(email)
            if not user:
                return jsonify({"success": False, "message": "该邮箱还未注册过账号", "action": "redirect",
                                "url": url_for('register')})

            code = ''.join(random.choices(string.digits, k=6))
            sender_email = "1587083264@qq.com"
            password = "muwrnhfxfvjfifdh"

            msg = EmailMessage()
            msg.set_content(
                f"您好！\n\n您正在疲劳驾驶智能检测系统进行密码重置操作。\n您的验证码是：【{code}】。\n该验证码将在 10 分钟后失效，请勿将验证码泄露给他人。")
            msg["Subject"] = "【疲劳驾驶检测系统】密码重置验证码"
            msg["From"] = f"安全中心 <{sender_email}>"
            msg["To"] = email

            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.qq.com", 465, context=context) as server:
                    server.login(sender_email, password)
                    server.send_message(msg)

                reset_codes[email] = {"code": code, "timestamp": time.time()}
                return jsonify({"success": True, "message": "验证码已成功发送到您的邮箱，请查收！"})
            except Exception as e:
                return jsonify({"success": False, "message": f"邮件发送失败: {str(e)}"})
        else:
            return jsonify({"success": False, "message": "请输入有效的注册邮箱！"})
    return render_template('forgot_password.html')


@app.route('/verify_code', methods=['POST'])
def verify_code():
    email = request.form.get('email')
    code = request.form.get('code')
    record = reset_codes.get(email)

    if record and record['code'] == code and time.time() - record['timestamp'] <= 600:
        session[f'verified_{email}'] = True
        return jsonify({"success": True})
    elif record and record['code'] != code:
        return jsonify({"success": False, "message": "验证码错误，请重新输入"})
    else:
        return jsonify({"success": False, "message": "验证码已失效，请重新获取"})


@app.route('/reset_with_code', methods=['POST'])
def reset_with_code():
    email = request.form.get('email')
    password = request.form.get('password')

    if not session.get(f'verified_{email}'):
        return jsonify({"success": False, "message": "验证流程无效，请重试"})

    if email and password:
        success, message = update_password(email, password)
        if success:
            session.pop(f'verified_{email}', None)
            reset_codes.pop(email, None)
            return jsonify(
                {"success": True, "message": "密码重置成功！请使用新密码重新登录...", "redirect_url": url_for('login')})
        else:
            return jsonify({"success": False, "message": message})
    else:
        return jsonify({"success": False, "message": "邮箱或新密码不能为空！"})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route("/")
@app.route('/index')
@login_required
def index():
    return render_template("index.html")


@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = get_all_users()
    return render_template('admin_dashboard.html', users=users)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user_route(user_id):
    success, message = delete_user(user_id)
    return jsonify({"success": success, "message": message})


@app.route('/admin/update_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def update_user_route(user_id):
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    success, message = update_user_info(user_id, username, email, password)
    return jsonify({"success": success, "message": message})


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session.get('user_id')
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        avatar_file = request.files.get('avatar')

        avatar_filename = None
        if avatar_file and avatar_file.filename != '':
            filename = f"avatar_{user_id}_{int(time.time())}.jpg"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            avatar_file.save(save_path)
            avatar_filename = filename
            session['avatar'] = filename

        if not username: username = session.get('username')
        success, message = update_user_info(user_id, username, email, password, avatar_filename)

        if success:
            session['username'] = username
            return jsonify({"success": True, "message": "个人信息更新成功"})
        else:
            return jsonify({"success": False, "message": message})

    user = get_user_by_id(user_id)
    return render_template('profile.html', user=user)


@app.route('/image-recognition')
@login_required
def image_recognition():
    return render_template('image_recognition.html')


@app.route('/video-recognition')
@login_required
def video_recognition():
    return render_template('video_recognition.html')


@app.route('/realtime-monitoring')
@login_required
def realtime_monitoring():
    return render_template('camera.html')


@app.route("/upload-image", methods=["POST"])
@login_required
def upload_image():
    if "file" not in request.files: return jsonify({"error": "未选择文件"})
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "未选择文件"})

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)

    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None: return jsonify({"error": "图像读取失败，请检查文件格式"})

    result = detector.predict(img, is_video=False)

    is_safe = (result['action_label'] == 0) and (result['face_label'] == 0)
    frontend_label = 0 if is_safe else 1

    class_text = f"行为: {result['action_zh']} | 面部: {result['face_zh']}"
    avg_conf = (result['action_conf'] + result['face_conf']) / 2 if result['face_conf'] > 0 else result['action_conf']

    if result["status"] != "no_object":
        action_text_color, _ = get_colors_by_label(result["action_label"] if result["action_label"] == 0 else 1)
        face_text_color, face_box_color = get_colors_by_label(result["face_label"])

        for (x, y, w, h) in result["faces"]:
            cv2.rectangle(img, (x, y), (x + w, y + h), face_box_color, 2)

        action_text = f"行为: {result['action_zh']}"
        face_text = f"面部: {result['face_zh']}"

        # 剔除了头部姿态的UI渲染
        img = put_chinese_text(img, action_text, (10, 20), action_text_color, font_size=32)
        img = put_chinese_text(img, face_text, (10, 65), face_text_color, font_size=32)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_base64}"

    try:
        os.remove(img_path)
    except:
        pass

    return jsonify({
        "class": class_text,
        "confidence": avg_conf,
        "label": frontend_label,
        "image_url": img_data_url
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/upload-video", methods=["POST"])
@login_required
def upload_video():
    if "file" not in request.files: return jsonify({"error": "未选择文件"})
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "未选择文件"})

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(video_path)

    filename = file.filename
    with buffer_lock:
        if filename in video_buffers:
            video_buffers[filename]['stop_event'].set()
            time.sleep(0.5)

        video_buffers[filename] = {
            'queue': queue.Queue(maxsize=300),
            'stop_event': threading.Event(),
            'fps': 30,
            'processing_fps': 0.0,
            'processed_count': 0,
            'current_play_frame': 0,
            'is_paused': False,
            'seek_target': None,
            'speed': 1.0,
            'total_frames': 0
        }

        t = threading.Thread(target=background_video_processor, args=(video_path, filename))
        t.daemon = True
        t.start()
        video_buffers[filename]['thread'] = t

    return render_template("video.html", video_filename=file.filename)


@app.route("/video-feed/<filename>")
@login_required
def video_feed(filename):
    return Response(video_stream_generator(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/video-state/<filename>")
@login_required
def get_video_state(filename):
    with buffer_lock:
        state = video_buffers.get(filename)
        if state:
            return jsonify({
                "status": "success",
                "data": {
                    "processed": state.get('processed_count'),
                    "total": state.get('total_frames'),
                    "current": state.get('current_play_frame'),
                    "paused": state.get('is_paused'),
                    "speed": state.get('speed'),
                    "fps": state.get('fps')
                }
            })
        return jsonify({"status": "error", "message": "Video not initialized yet"})


@app.route("/video-control/<filename>", methods=["POST"])
@login_required
def video_control(filename):
    action = request.json.get("action")
    value = request.json.get("value")

    with buffer_lock:
        if filename not in video_buffers:
            return jsonify({"status": "error", "message": "Video not found"})

        if action == "stop":
            video_buffers[filename]['stop_event'].set()
        elif action == "pause":
            video_buffers[filename]['is_paused'] = True
        elif action == "play":
            video_buffers[filename]['is_paused'] = False
        elif action == "set_speed":
            try:
                video_buffers[filename]['speed'] = float(value)
            except:
                pass
        elif action == "seek":
            try:
                target_percent = float(value)
                total = video_buffers[filename]['total_frames']
                target_frame = int(total * (target_percent / 100.0))
                video_buffers[filename]['seek_target'] = target_frame
                video_buffers[filename]['current_play_frame'] = target_frame
                video_buffers[filename]['is_paused'] = False
            except:
                pass

    return jsonify({"status": "success", "message": "Control signal sent"})


@app.route("/camera-feed")
@login_required
def camera_feed():
    def camera_detection():
        cap = cv2.VideoCapture(0)
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret: break
            try:
                result = detector.predict(frame, is_video=True, fps=cam_fps)

                if result["status"] == "no_object":
                    pass
                else:
                    action_text_color, _ = get_colors_by_label(
                        result["action_label"] if result["action_label"] == 0 else 1)
                    face_text_color, face_box_color = get_colors_by_label(result["face_label"])

                    action_text = f"身体行为: {result['action_zh']} ({result['action_conf']:.2f})"
                    face_text = f"面部状态: {result['face_zh']} ({result['face_conf']:.2f})"

                    # 剔除了头部姿态的UI渲染
                    frame = put_chinese_text(frame, action_text, (10, 20), action_text_color, font_size=32)
                    frame = put_chinese_text(frame, face_text, (10, 65), face_text_color, font_size=32)

                    for (x, y, w, h) in result["faces"]:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), face_box_color, 2)

            except Exception:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(camera_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin/debug-image')
@login_required
@admin_required
def admin_debug_image():
    return render_template('admin_debug_image.html')


@app.route("/admin/upload-debug-image", methods=["POST"])
@login_required
@admin_required
def upload_debug_image():
    if "file" not in request.files: return jsonify({"error": "未选择文件"})
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "未选择文件"})

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"debug_{int(time.time())}_{file.filename}")
    file.save(img_path)

    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if img is None:
        try:
            os.remove(img_path)
        except:
            pass
        return jsonify({"error": "图像读取失败"})

    annotated_bytes, info = detector.predict_with_landmarks(img)

    img_base64 = base64.b64encode(annotated_bytes).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_base64}"

    try:
        os.remove(img_path)
    except:
        pass

    return jsonify({
        "status": "success",
        "image_url": img_data_url,
        "media_pipe_info": info
    })


@app.route('/admin/debug-video')
@login_required
@admin_required
def admin_debug_video():
    return render_template('admin_debug_video.html')


@app.route("/admin/upload-debug-video", methods=["POST"])
@login_required
@admin_required
def upload_debug_video():
    if "file" not in request.files: return jsonify({"error": "未选择文件"})
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "未选择文件"})

    filename = f"debug_{int(time.time())}_{file.filename}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    with buffer_lock:
        if filename in video_buffers:
            video_buffers[filename]['stop_event'].set()
            time.sleep(0.5)

        video_buffers[filename] = {
            'queue': queue.Queue(maxsize=300),
            'stop_event': threading.Event(),
            'fps': 30,
            'processed_count': 0
        }

        t = threading.Thread(target=background_debug_video_processor, args=(video_path, filename))
        t.daemon = True
        t.start()
        video_buffers[filename]['thread'] = t

    return render_template("admin_debug_video.html", video_filename=filename)


@app.route("/admin/debug-video-feed/<filename>")
@login_required
@admin_required
def debug_video_feed(filename):
    return Response(debug_video_stream_generator(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    from waitress import serve

    print("正在启动服务器...")
    print("服务运行在 http://127.0.0.1:5000")
    serve(app, host="0.0.0.0", port=5000, threads=8)