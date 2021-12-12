from PIL import Image
import tensorflow.compat.v1 as tf
import cv2
import numpy as np
import os
import hashlib


def expand_bbox_to_square(bbox_coords):
    # bbox_coords = [xmin, ymin, xmax, ymax]
    w = bbox_coords[2] - bbox_coords[0]
    h = bbox_coords[3] - bbox_coords[1]
    x_center = bbox_coords[0] + w / 2
    y_center = bbox_coords[1] + h / 2

    bbox_coords_new = [0, 0, 0, 0]
    if w < h:
        new_w = h
        bbox_coords_new[0] = x_center - new_w / 2
        bbox_coords_new[1] = bbox_coords[1]
        bbox_coords_new[2] = x_center + new_w / 2
        bbox_coords_new[3] = bbox_coords[3]
    elif w > h:
        new_h = w
        bbox_coords_new[0] = bbox_coords[0]
        bbox_coords_new[1] = y_center - new_h / 2
        bbox_coords_new[2] = bbox_coords[2]
        bbox_coords_new[3] = y_center + new_h / 2
    else:
        bbox_coords_new[0] = bbox_coords[0]
        bbox_coords_new[1] = bbox_coords[1]
        bbox_coords_new[2] = bbox_coords[2]
        bbox_coords_new[3] = bbox_coords[3]


    return np.array(bbox_coords_new, dtype=np.int)


def img_read(im_path):
    """
    cv2 open image
    :param im_path:
    :return:
    """
    try:
        im = cv2.imread(im_path)
    except Exception as e:
        print('Broken image:', im_path)
        print(e)
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return im


def image_processing(im: np.array,
                     face_xywh=(0, 0, 0, 0),
                     margin: float = 0,
                     square: bool = True,
                     convert_RGB: bool = True):
    """
    cv2 image processing
    :param im: np.array
    :param face_xywh:  (x, y, w, h) bbox in pixels. (0, 0, 0, 0) if unknown.
    :param margin: margin in percents 0..100%
    :param square:
    :param convert_RGB: bool
    :return:
    """
    assert 0 <= margin <= 1
    if convert_RGB and (len(im.shape) == 3):
        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_height, im_width = im.shape[:2]
    if face_xywh == (0, 0, 0, 0):
        face_xywh = (0, 0, im_width, im_height)  # if unknown GT boxes, use all image.
    if square:
        x, y, x2, y2 = expand_bbox_to_square((face_xywh[0],
                                              face_xywh[1],
                                              face_xywh[0] + face_xywh[2],
                                              face_xywh[1] + face_xywh[3]))
        w, h = x2 - x, y2 - y
    else:
        x, y, w, h = face_xywh
    x, y, w, h = int(x - margin * w), int(y - margin * h), int(w + margin * w), int(h + margin * h)
    if x < 0:
        im = cv2.copyMakeBorder(im, 0, 0, np.abs(x), np.abs(x), borderType=0, value=[0, 0, 0])
        x = 0
    if y < 0:
        im = cv2.copyMakeBorder(im, np.abs(y), np.abs(y), 0, 0, borderType=0, value=[0, 0, 0])
        y = 0
    im = im[y:y + h, x:x + w]
    return im


# def image_crop_with_padding(img: np.array, bbox: np.array) -> np.array:
#     """
#     Cropping image with padding, if coordinates bbox <> image.size
#     :param img: cv2 image
#     :param bbox: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
#     :return:
#     """
#     h, w = img.shape[:2]
#     tlbr = bbox.copy()
#     tlx, tly = tlbr[:2]
#     brx, bry = tlbr[2:]
#     pad_ly, pad_ry, pad_lx, pad_rx = (0, 0, 0, 0)
#     if tlx < 0:
#         pad_lx = -tlx
#         tlx = 0
#     if tly < 0:
#         pad_ly = -tly
#         tly = 0
#     if brx > w:
#         pad_rx = brx-w
#         brx = w
#     if bry > h:
#         pad_ry = bry-h
#         bry = h
#     crop_img = img[tly:bry, tlx:brx]
#     crop_img_pad = cv2.copyMakeBorder(crop_img, pad_ly, pad_ry, pad_lx, pad_rx, borderType=0, value=[0, 0, 0])
#     return crop_img_pad


def image_crop_with_padding(img: np.array, bbox: np.array) -> np.array:
    """
    Cropping image with padding, if coordinates bbox <> image.size
    :param img: cv2 image
    :param bbox: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    :return:
    """
    h, w = img.shape[:2]
    tlbr = bbox.copy()
    tlx, tly = tlbr[:2]
    brx, bry = tlbr[2:]
    pad_ly, pad_ry, pad_lx, pad_rx = (0, 0, 0, 0)
    if tlx < 0:
        pad_lx += np.abs(tlx)/2
        pad_rx += np.abs(tlx)/2
        tlx = 0
    if tly < 0:
        pad_ly += np.abs(tly) / 2
        pad_ry += np.abs(tly) / 2
        tly = 0
    if brx > w:
        pad_lx += np.abs(brx-w) / 2
        pad_rx += np.abs(brx-w) / 2
        brx = w
    if bry > h:
        pad_ly += np.abs(bry-h) / 2
        pad_ry += np.abs(bry-h) / 2
        bry = h
    crop_img = img[tly:bry, tlx:brx]
    # crop_img = img.copy()
    crop_img_pad = cv2.copyMakeBorder(crop_img, int(pad_ly), int(pad_ry), int(pad_lx), int(pad_rx), borderType=0, value=[0, 0, 0])
    return crop_img_pad

# noinspection PyUnresolvedReferences
def init_pb(model_path):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    compute_graph = tf.Graph()
    compute_graph.as_default()
    sess = tf.Session(config=config)
    with tf.gfile.GFile(model_path, 'rb') as fid:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fid.read())
        tf.import_graph_def(graph_def, name='')
    return compute_graph, sess


def img_save(image: np.array, img_path: str) -> bool:
    """
    Save PIL image to jpg if img_path dont exist
    :param image:
    :param img_path:
    :return: bool
    """
    if not os.path.exists(img_path):
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return True


def img_to_hash(image: np.array):
    """
    Get hash from Image
    :param image:
    :return:
    """
    img_hash = hashlib.sha256(image.tobytes())
    return img_hash.hexdigest()


def open_video_stream(source: str = None):
    """
    Open videofile or videostrem
    :param source:
    :return: (True, cv2.VideoCapture) or (False, str)
    """
    if not source:
        return False, 'sourse is empty'
    try:
        return True, cv2.VideoCapture(source)
    except Exception as e:
        print("videostream don't open")
        return False, e.__str__()


def draw_rect(img, bbox, text: str = ""):
    """
    Draw rectangle with text in image
    :param img:
    :param bbox: (xtl,ytl, xbr, ybr)
    :param text:
    :return:
    """
    xtl, ytl, xbr, ybr = bbox
    img = cv2.rectangle(img.copy(), (int(xtl), int(ytl)), (int(xbr), int(ybr)), (0, 255, 0), 2)
    img = cv2.putText(img, text, (int(xtl-10), int(ytl - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

