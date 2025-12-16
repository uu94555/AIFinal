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
STABILITY_WINDOW = 7  # 穩定度視窗大小 (越大越穩但越慢)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 設定字型路徑 (Windows 預設)
font_path_tw = "C:/Windows/Fonts/msjh.ttc"       # 微軟正黑體
font_path_emoji = "C:/Windows/Fonts/seguiemj.ttf" # Windows Emoji 字型

# 嘗試載入字型，若失敗則使用預設 (避免程式崩潰)
try:
    font_tw = ImageFont.truetype(font_path_tw, 30)
    font_emoji = ImageFont.truetype(font_path_emoji, 60)
except IOError:
    print("⚠️ 警告：找不到指定字型，將使用系統預設字型 (中文/Emoji 可能無法正常顯示)")
    font_tw = ImageFont.load_default()
    font_emoji = ImageFont.load_default()

# --- 截圖相關設定 ---
SNAPSHOT_FOLDER = 'snapshots'
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# --- 全域變數 ---
global_frame = None       # 用來暫存最新畫面供截圖用
lock = threading.Lock()   # 執行緒鎖，確保讀寫安全

# --- 類別定義：飄浮 Emoji ---
class FloatingEmoji:
    def __init__(self, emoji_char, start_x, start_y):
        self.char = emoji_char
        self.x = start_x
        self.y = start_y
        self.speed = random.uniform(3, 7)    # 上升速度
        self.drift = random.uniform(-1, 1)   # 左右飄移
        
    def update(self):
        self.y -= self.speed
        self.x += self.drift
        
    def is_off_screen(self):
        return self.y < -50 # 超出上方邊界

    def draw(self, draw_obj):
        draw_obj.text((self.x, self.y), self.char, font=font_emoji, fill=(255, 255, 255))

# --- 初始化模型與變數 ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
particles = []
emotion_queue = deque(maxlen=STABILITY_WINDOW)
spawn_timer = 0

# 情緒腳本設定
emotion_script = {
    'sad':      {'msg': "今天怎麼了？呼呼", 'emoji': "😢", 'color': (100, 149, 237)},
    'happy':    {'msg': "看起來心情很好喔！", 'emoji': "😄", 'color': (255, 105, 180)},
    'angry':    {'msg': "深呼吸... 別生氣",   'emoji': "😤", 'color': (255, 69, 0)},
    'neutral':  {'msg': "保持平靜...",          'emoji': None, 'color': (200, 200, 200)},
    'surprise': {'msg': "哇！嚇到了嗎？",      'emoji': "😲", 'color': (255, 215, 0)},
    'fear':     {'msg': "別怕，我在這",        'emoji': "😱", 'color': (148, 0, 211)},
    'disgust':  {'msg': "不喜歡嗎？",          'emoji': "🤢", 'color': (50, 205, 50)}
}

def get_most_frequent_emotion(queue):
    if not queue: return None
    return Counter(queue).most_common(1)[0][0]

# --- 核心邏輯：影像產生器 ---
def generate_frames():
    global particles, spawn_timer, global_frame, emotion_queue
    
    cap = cv2.VideoCapture(0)
    # 強制設定解析度，確保效能與繪圖位置正確
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. 鏡像翻轉 (讓操作更直覺)
        frame = cv2.flip(frame, 1)

        # 2. 人臉偵測 (針對乾淨的原始畫面)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            emotion_queue.clear()

        detected_emotion_for_spawn = None
        face_draw_info = [] # 暫存要畫的資訊，稍後統一畫

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                # DeepFace 分析
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                
                # 加入穩定佇列
                emotion_queue.append(result[0]['dominant_emotion'])
                current_stable_emotion = get_most_frequent_emotion(emotion_queue)
                
                detected_emotion_for_spawn = current_stable_emotion

                # 準備繪圖資訊
                if current_stable_emotion in emotion_script:
                    script = emotion_script[current_stable_emotion]
                    face_draw_info.append({
                        'rect': (x, y, w, h),
                        'msg': script['msg'],
                        'color': script['color']
                    })
            except Exception:
                pass

        # 3. PIL 繪圖處理 (開始在畫面上加料)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # A. 處理 Emoji 粒子產生
        if detected_emotion_for_spawn and emotion_script[detected_emotion_for_spawn]['emoji']:
            target_emoji = emotion_script[detected_emotion_for_spawn]['emoji']
            spawn_timer += 1
            if spawn_timer > 5: # 控制產生頻率
                start_x = random.randint(50, FRAME_WIDTH - 50)
                start_y = FRAME_HEIGHT + 10
                particles.append(FloatingEmoji(target_emoji, start_x, start_y))
                spawn_timer = 0

        # B. 更新並繪製所有粒子
        for p in particles:
            p.update()
            p.draw(draw)
        # 清除超出畫面的粒子
        particles = [p for p in particles if not p.is_off_screen()]

        # C. 繪製人臉框與文字
        for info in face_draw_info:
            x, y, w, h = info['rect']
            color = info['color']
            # 畫框 (PIL 座標: 左上x, 左上y, 右下x, 右下y)
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            # 畫字
            draw.text((x, y - 40), info['msg'], font=font_tw, fill=color)

        # 4. 轉回 OpenCV 格式
        final_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 5. 更新全域變數 (供截圖用，需上鎖)
        with lock:
            global_frame = final_frame.copy()

        # 6. 編碼成 JPEG 串流傳給網頁
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask 路由設定 ---

@app.route('/')
def index():
    """顯示主網頁"""
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
        # 產生檔名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOT_FOLDER, filename)
        
        # 存檔 (使用鎖確保安全)
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
    # host='0.0.0.0' 讓區域網路內的其他裝置也能連線
    app.run(debug=True, host='0.0.0.0', port=5000)