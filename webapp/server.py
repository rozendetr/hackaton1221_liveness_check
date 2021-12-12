from flask import Flask, render_template, request, Response

# import face_recognition
import cv2
import numpy as np
import base64 # для перевода из формата 
from PIL import Image
from io import BytesIO
from src.pipeline import *


# from pyngrok import conf, ngrok
# if False: # если будет лень запускать ngrok из командной строки
#     conf.get_default().region = "eu"
#     http_tunnel = ngrok.connect(5000, 'http')
#     print(http_tunnel)




class VideoCamera(object): #TODO  возможно не понадобится
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('view.html')

@app.route('/video_feed') #TODO  возможно не понадобится
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def draw_result(frame, res_predict):

    if not res_predict:
        return None
    real_h, real_w = frame.shape[:2]
    mask_frame = np.zeros((real_h, real_w, 3), dtype=np.uint8)
    for face_id, face_coord in enumerate(res_predict):
        print(face_coord)
        x_tl, y_tl, x_br, y_br, class_id = face_coord
        w_bbox = x_br-x_tl
        h_bbox = y_br-y_tl

        if class_id == 0:
            mask_frame[y_tl:y_br, x_tl:x_br, 1] = 255
        else:
            mask_frame[y_tl:y_br, x_tl:x_br, 0] = 255
    img_union = cv2.addWeighted(frame, 1, mask_frame, 0.7, 0.0)
    return img_union


@app.route('/submit', methods=['POST'])
def submit():
    content = request.get_json()
    # print(content)
    # image_str = request.args.get('image').replace(' ', '+')
    image_str = content.get("data").replace(' ', '+')
    image_str = image_str.split(',', maxsplit=1)[1]
    image_bytes = base64.b64decode(image_str.encode('ascii'))
    # with open("screen.png", "wb") as f: f.write(image_bytes)
    im = Image.open(BytesIO(image_bytes))
    cv_image = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
    res = nn_pipeline.predict_images([cv_image])
    if res:
        pred_img = draw_result(cv_image, res[0])

    # print(content.get("frame_id"))
    # cv2.imwrite("screen2.jpg", cv_image)
    return ""

if __name__ == "__main__":
    detector = SSD(pb_path=r"./frozen_inference_graph.pb", input_res=(640, 640), detector_margin=0.0)
    spoof_classificator = LivenessSpoof(weights=r"./2021-12-12-04-19_Anti_Spoofing_4_0_0_224x224_model_iter-12500.onnx")
    nn_pipeline = NNPipeLine(detector=detector,
                             spoof_classificator=spoof_classificator)
    app.run()
