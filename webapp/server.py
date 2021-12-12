from flask import Flask, render_template, request, Response

# import face_recognition
import cv2
import numpy as np
import base64 # для перевода из формата 
from PIL import Image
from io import BytesIO

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
    return render_template('view2.html')

@app.route('/video_feed') #TODO  возможно не понадобится
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    # print(cv_image.shape)
    # cv2.imwrite("screen2.jpg", cv_image)
    return ""

if __name__ == "__main__":
    app.run()