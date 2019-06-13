from PIL import Image
import cv2
import numpy as np


def trans_from_cv2_to_PIL(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return img


def trans_from_PIL_to_cv2(pil_img):
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
