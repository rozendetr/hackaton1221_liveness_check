import face_recognition
import cv2

from flask import Flask, render_template, request, Response

import numpy as np
import base64
import dash
import sys

from pyngrok import conf, ngrok
if False:
    conf.get_default().region = "eu"
    http_tunnel = ngrok.connect(5000, 'http')
    print(http_tunnel)

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

@app.route('/submit',methods=['POST'])
def submit():
    image_str = request.args.get('image').split(',')[1]
    print(image_str)
    image_bin = base64.b64decode(image_str)
    
    # что-то не получается картинка
    with open("screen.png", "wb") as f:
        f.write(image_bin)

    # пока не уверен что так надо
    image = np.asarray(bytearray(image_bin), dtype="uint8")
    #print(image)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return ""

if __name__ == "__main__":
    app.run()