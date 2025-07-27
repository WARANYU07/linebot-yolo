import os
import cv2
import numpy as np
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, ImageMessageContent
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent as WebhookImageMessageContent

# --- 1. กรอกข้อมูลของคุณ ---
LINE_CHANNEL_ACCESS_TOKEN = "ldGBgN9pEvNUsLxwK/khO2jCnaPbrS45zgRo0wWYKWjNDlsQieFEmGsGrz12LO6hxe6+D+sfoLqXm4um7GO6s/8vvcHgLxtpkDQGoTDL0vXldT91gIzBXUW7eRtHefC6h5Weu2KcpNGbLBbE8P3fOQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "77d8cb2ef1de79dbc62ce28771af0ad9"
# --- สิ้นสุดส่วนที่ต้องแก้ไข ---

# --- 2. ตั้งค่าโมเดล YOLO ---
YOLO_CONFIG = 'train.cfg'
YOLO_WEIGHTS = 'train.weights'
YOLO_NAMES = 'names.list'

# โหลดโมเดล YOLO
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
classes = []
with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# --- สิ้นสุดส่วนตั้งค่าโมเดล ---

app = Flask(__name__)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/webhook", methods=['POST'])
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=WebhookImageMessageContent)
def handle_image_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        # ดึงข้อมูลรูปภาพจาก LINE
        message_content = line_bot_api.get_message_content(message_id=event.message.id)
        
        # แปลงรูปภาพเพื่อใช้กับ OpenCV
        img_array = np.frombuffer(message_content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # ประมวลผลภาพด้วย YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # หาผลลัพธ์
        class_ids = []
        confidences = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: # ตั้งค่าความมั่นใจขั้นต่ำที่ 50%
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        # สร้างข้อความตอบกลับ
        if len(class_ids) > 0:
            # เรียงลำดับตามความมั่นใจสูงสุด
            highest_confidence_index = np.argmax(confidences)
            predicted_class = classes[class_ids[highest_confidence_index]]
            confidence_percent = confidences[highest_confidence_index] * 100
            reply_text = f"ผลการวิเคราะห์: {predicted_class}\nความมั่นใจ: {confidence_percent:.2f}%"
        else:
            reply_text = "ไม่สามารถตรวจจับวัตถุในภาพได้"

        # ส่งข้อความตอบกลับไปยังผู้ใช้
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
     with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="กรุณาส่งรูปภาพเพื่อวิเคราะห์ครับ")]
            )
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)