import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque, Counter
import random
import datetime
import threading
import os
from flask import Flask, render_template, Response, jsonify, send_from_directory

app = Flask(__name__)

# --- 設定區 ---
STABILITY_WINDOW = 7
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- emoji圖片設定 ---
IMAGE_FOLDER = 'img'
IMAGE_SIZE = (60, 60)

# 設定字型路徑
font_path_tw = "C:/Windows/Fonts/msjh.ttc"       # 微軟正黑體

# 嘗試載入字型，若失敗則使用預設
try:
    font_tw = ImageFont.truetype(font_path_tw, 30)
except IOError:
    print("⚠️警告：找不到指定字型，將使用系統預設字型 ")
    font_tw = ImageFont.load_default()
    
# --- 拍照相關設定 ---
SNAPSHOT_FOLDER = 'snapshots'
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# --- 全域變數 ---
global_frame = None       # 用來暫存最新畫面供截圖用
lock = threading.Lock()   # 執行緒鎖，確保讀寫安全

# --- 飄浮圖片粒子 ---
class FloatingImageParticle:
    def __init__(self, image_filename, start_x, start_y):
        
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        self.valid = True
        
        # 1. 圖片轉rgb
        img = Image.open(image_path).convert("RGBA") 
        self.image = img.resize(IMAGE_SIZE)
            
        # 2. 初始化位置和速度
        self.x = start_x
        self.y = start_y
        self.speed = random.uniform(3, 7)    # 上升
        self.drift = random.uniform(-1, 1)   # 左右
        self.half_width = IMAGE_SIZE[0] / 2

    def update(self):
        self.y -= self.speed
        self.x += self.drift
        
    def is_off_screen(self):
        return self.y < -50

# --- 初始化模型與變數 ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
particles = []
emotion_queue = deque(maxlen=STABILITY_WINDOW)
spawn_timer = 0

emotion_script = {
    'sad':      {'msg': "今天怎麼了？呼呼",'image_file': "sad.png", 'color': (100, 149, 237)},
    'happy':    {'msg': "心情很好喔！", 'image_file': "happy.png", 'color': (255, 105, 180)},
    'angry':    {'msg': "深呼吸... 別生氣", 'image_file': "angry.png", 'color': (255, 69, 0)},
    'neutral':  {'msg': "保持平靜...", 'image_file': None, 'color': (122, 122, 122)},
    'surprise': {'msg': "哇！嚇到了嗎？", 'image_file': "surprise.png", 'color': (255, 215, 0)},
    'fear':     {'msg': "別怕，我在這", 'image_file': "fear.png", 'color': (148, 0, 211)},
    'disgust':  {'msg': "不喜歡嗎？", 'image_file': "disgust.png", 'color': (50, 205, 50)}
}

def get_most_frequent_emotion(queue):
    if not queue: return None
    return Counter(queue).most_common(1)[0][0]

# --- 影像產生器 ---
def generate_frames():
    global particles, spawn_timer, global_frame, emotion_queue
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. 鏡像翻轉
        frame = cv2.flip(frame, 1)

        # 2. 人臉偵測與 DeepFace 分析
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            emotion_queue.clear()

        detected_emotion_for_spawn = None
        face_draw_info = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                # 執行 DeepFace 分析，silent=True 減少控制台輸出
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                
                emotion_queue.append(result[0]['dominant_emotion'])
                current_stable_emotion = get_most_frequent_emotion(emotion_queue)
                
                detected_emotion_for_spawn = current_stable_emotion

                if current_stable_emotion in emotion_script:
                    script = emotion_script[current_stable_emotion]
                    face_draw_info.append({
                        'rect': (x, y, w, h),
                        'msg': script['msg'],
                        'color': script['color']
                    })
            except Exception:
                pass

# 3. PIL 繪圖處理
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 處理圖片粒子產生
        script = emotion_script.get(detected_emotion_for_spawn)
        if script and script['image_file']:
            target_image = script['image_file']
            spawn_timer += 1
            if spawn_timer > 5: # 控制產生頻率
                start_x = random.randint(50, FRAME_WIDTH - 50)
                start_y = FRAME_HEIGHT + 10
                particles.append(FloatingImageParticle(target_image, start_x, start_y))
                spawn_timer = 0

        new_particles = []
        
        # 臉部邊界
        face_rects = [info['rect'] for info in face_draw_info] 

        for p in particles:
            p.update()
            
            is_colliding = False
            
            # 粒子圖片邊界
            p_x1 = int(p.x - p.half_width)
            p_y1 = int(p.y)
            p_x2 = p_x1 + IMAGE_SIZE[0]
            p_y2 = p_y1 + IMAGE_SIZE[1]
            
            # 碰撞偵測
            for (fx, fy, fw, fh) in face_rects:
                if (p_x1 < fx + fw and
                    p_x2 > fx and
                    p_y1 < fy + fh and
                    p_y2 > fy):
                    is_colliding = True
                    break
            
            if not p.is_off_screen() and p.valid and not is_colliding:
                paste_x = p_x1
                paste_y = p_y1
                img_pil.paste(p.image, (paste_x, paste_y), p.image) 
                new_particles.append(p)

        particles = new_particles

        # 文字
        for info in face_draw_info:
            x, y, w, h = info['rect']
            color = info['color']
            draw.text((x, y - 40), info['msg'], font=font_tw, fill=color)

        final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with lock:
            global_frame = final_frame.copy()

        # 串流傳給網頁
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask 路由設定 ---

@app.route('/')
def index():
    """顯示主網頁 (假設 templates/index.html 存在)"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """提供影像串流"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    """處理截圖請求"""
    global global_frame
    if global_frame is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOT_FOLDER, filename)
        
        with lock:
            cv2.imwrite(filepath, global_frame)
            
        print(f"✅ 已截圖並儲存: {filepath}")
        return jsonify({"status": "success", "filename": filename})
    else:
        return jsonify({"status": "error", "message": "No frame available"})

@app.route('/snapshots/<filename>')
def get_snapshot_file(filename):
    """讓前端可以讀取截圖檔案"""
    return send_from_directory(SNAPSHOT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)