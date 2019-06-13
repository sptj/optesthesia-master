import numpy as np
import cv2
import os


def video_transfer(src_path, dst_path):
    if not os.path.isfile(src_path):
        print('source file not found')
        exit(0)
    dst_parentdir, dst_filename = os.path.split(dst_path)
    if not os.path.exists(dst_parentdir):
        # make multi level dir
        os.mkdirs(dst_parentdir)
    video_capture = cv2.VideoCapture(src_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    fourcc_specific = {
        '.mp4': 'MJPG',
        '.avi': 'XVID',
        '.ogv': 'THEO',
        '.flv': 'FLV1',
        '.wmv': 'MJPG',
        '.mkv': '3IVX',
    }

    (_, file_ext) = os.path.splitext(dst_filename)
    file_ext = file_ext.lower()
    if file_ext not in fourcc_specific:
        print('dist format not support')
        exit(0)
    fourcc_type = fourcc_specific[file_ext]
    video_writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*fourcc_type), fps, size)
    status, frame = video_capture.read()
    i = 0
    while status:
        i = i + 1
        video_writer.write(frame)
        status, frame = video_capture.read()
        if (i % 100 == 0):
            print('processed', int(i / frame_count * 100), '%')
    video_writer.release()
    video_capture.release()
