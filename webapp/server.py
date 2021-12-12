
from flask import Flask, render_template, request, Response, send_file


# import face_recognition
import cv2
import numpy as np
import base64 # для перевода из формата 
from PIL import Image
from io import BytesIO
from src.pipeline import *
import time


# from pyngrok import conf, ngrok
#
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
    #while True:
    #    frame = camera.get_frame()
    #    yield (b'--frame\r\n'
    #           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    print()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('view.html')

@app.route('/video_feed') #TODO  возможно не понадобится
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#cv_image = ''

def draw_result(frame, res_predict, proc_time):

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
            mask_frame[y_tl:y_br, x_tl:x_br, 2] = 255
    img_union = cv2.addWeighted(frame, 1, mask_frame, 0.5, 0.0)
    cv2.putText(img_union, str(proc_time),
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),
                1,
                2)
    return img_union

pred_img = None

def generate_image():
    """Get image from url. Read to Pillow. Draw. Save in BytesIO"""
    # read to pillow
    image = Image.open("screen2.jpg")  #
    # if image:
    #     print(image.size)
    # image =  Image.fromarray(pred_img.astype('uint8'))
    # convert to file-like data
    obj = BytesIO()             # file in memory to save image without using disk  #
    image.save(obj, format='png')  # save in file (BytesIO)
    obj.seek(0)                    # move to beginning of file (BytesIO) to read it   #
    return obj

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        content = request.get_json()
        # print(content)
        # image_str = request.args.get('image').replace(' ', '+')
        image_str = content.get("data").replace(' ', '+')
        image_str = image_str.split(',', maxsplit=1)[1]
        image_bytes = base64.b64decode(image_str.encode('ascii'))
        # with open("screen.png", "wb") as f: f.write(image_bytes)
        im = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        start_time = time.time()
        res = nn_pipeline.predict_images([cv_image])
        proc_time = time.time() - start_time
        proc_time = round(proc_time, 3)
        if res:
            pred_img = draw_result(cv_image, res[0], proc_time)
            cv2.imwrite("screen2.jpg", pred_img)
            # cv2.imwrite("pred_img.jpg", pred_img)
            # im_pil = Image.fromarray(pred_img)
            # file_object = BytesIO()
            # im_pil.save(file_object, 'PNG')  # save as PNG in file in memory
            # file_object.seek(0)
            # return send_file(file_object, mimetype='image/png')

            # print(content.get("frame_id"))
        # cv2.imwrite("screen2.jpg", cv_image)
        return ""


@app.route('/get_image', methods=['GET'])
def get_image():
    file_object = generate_image()
    # img = Image.open("screen.jpg")
    # print(type(img))
    # img = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2GRAY)
    # print(type(cv_image))
    # file_object = BytesIO()         # create file-object in memory
    # img.save(file_object, 'JPEG')    # write JPG in file-object
    # # move to beginning of file so `send_file()` it will read from start
    # file_object.seek(0)
    return send_file(file_object, mimetype='image/jpeg')




if __name__ == "__main__":
    detector = SSD(pb_path=r"./frozen_inference_graph.pb", input_res=(640, 640), detector_margin=0.0)
    spoof_classificator = LivenessSpoof(weights=r"./2021-12-12-04-19_Anti_Spoofing_4_0_0_224x224_model_iter-12500.onnx")
    nn_pipeline = NNPipeLine(detector=detector,
                             spoof_classificator=spoof_classificator)
    app.run()
