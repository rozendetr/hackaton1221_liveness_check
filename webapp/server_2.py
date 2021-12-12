from flask import Flask, render_template, request, Response, send_file

# import face_recognition
import cv2
import numpy as np
import base64 # для перевода из формата
from PIL import Image
from io import BytesIO
from src.pipeline import *


app = Flask(__name__)



@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<body>
<canvas></canvas>
<script>
var canvas = document.getElementsByTagName('canvas');
var ctx = canvas[0].getContext('2d');

var img = new Image();
img.src = "/api/b";

// it can't draw it at once. it has to wait till it is loaded
//ctx.drawImage(img, 0, 0);

img.onload = function() {
  img.style.display = 'none'; // I don't know why they hide it

  console.log('WxH: ' + img.width + 'x' + img.height)

  // convert Image to ImageData
  //(it has to draw on canvas so it could need second canvas for this)

  ctx.drawImage(img, 0, 0);
  var imageData = ctx.getImageData(0, 0, img.width, img.height)
  ctx.putImageData(imageData, x, y);
  //put ImageData many times  
  
};
</script>
</body>
</html>
'''


def generate_image():
    """Get image from url. Read to Pillow. Draw. Save in BytesIO"""

    # read to pillow
    image = Image.open("screen2.jpg")  #
    # convert to file-like data
    obj = BytesIO()             # file in memory to save image without using disk  #
    image.save(obj, format='png')  # save in file (BytesIO)                           # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    obj.seek(0)                    # move to beginning of file (BytesIO) to read it   #

    return obj

@app.route('/api/b')
def array():
    '''
    generate image from numpy.array using PIL.Image
    and send without saving on disk using io.BytesIO'''

    # arr = np.array([
    #     [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
    #     [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255, 255,   0, 255, 255,   0, 255, 255,   0],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255, 255, 255, 255, 255, 255, 255, 255,   0],
    #     [  0, 255,   0, 255, 255, 255, 255,   0, 255,   0],
    #     [  0, 255, 255,   0,   0,   0,   0, 255, 255,   0],
    #     [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
    #     [255, 255,   0,   0,   0,   0,   0,   0, 255, 255],
    # ])
    # arr = cv2.imread("./pred_img.jpg")
    # img = Image.open("./screen2.jpg")
    # # img = Image.fromarray(arr.astype('uint8')) # convert arr to image
    # print('size', img.size)
    # file_object = BytesIO()   # create file in memory
    # img.save(file_object, 'PNG') # save as PNG in file in memory
    # file_object.seek(0)          # move to beginning of file
    #                              # so send_file() will read data from beginning of file

    file_object = generate_image()

    return send_file(file_object,  mimetype='image/png')


if __name__ == "__main__":
    # detector = SSD(pb_path=r"./frozen_inference_graph.pb", input_res=(640, 640), detector_margin=0.0)
    # spoof_classificator = LivenessSpoof(weights=r"./2021-12-12-04-19_Anti_Spoofing_4_0_0_224x224_model_iter-12500.onnx")
    # nn_pipeline = NNPipeLine(detector=detector,
    #                          spoof_classificator=spoof_classificator)
    app.run()