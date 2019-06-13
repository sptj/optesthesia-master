import os
import sys

import cv2
from pascal_voc_io import PascalVocWriter


def annotation_video(video_path):
    with VideoCap