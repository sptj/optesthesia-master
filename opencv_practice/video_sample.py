import numpy as np
import cv2
import os
from time import sleep

def video_sample(src_path, dst_path):
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

    status, frame = video_capture.read()
    i = 0
    isRecording = False
    isPlaying = False
    while status:
        img2show = frame.copy()
        if isRecording:
            cv2.putText(img2show, 'Video is Recording', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        else:
            cv2.putText(img2show, 'Video is not Recording', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow('sampling...', img2show)
        key = cv2.waitKey(1) & 0xff
        if key == ord(' '):
            if isRecording == False:
                i = i + 1
                dst_filename_bld, dst_filename_ext = os.path.splitext(dst_path)
                sec_filename = '{}_patch_{}{}'.format(dst_filename_bld, i, dst_filename_ext)
                video_writer = cv2.VideoWriter(sec_filename, cv2.VideoWriter_fourcc(*fourcc_type), fps, size)
                isRecording = True
                print('isRecording == False')
            elif isRecording == True:
                video_writer.release()
                isRecording = False
                print('isRecording == True')
        elif key == 13:
            isPlaying = not isPlaying
        if isRecording and isPlaying:
            video_writer.write(frame)
        else:
            sleep(0.02)
        if isPlaying:
            status, frame = video_capture.read()
    try:
        video_writer.release()
    except:
        print('unexcept quit')
    video_capture.release()


if __name__ == '__main__':
    video_sample(r'G:\VIDEO_TO_ANNO\Introducing the DJI Mavic 2.mp4', r'G:\VIDEO_TO_ANNO\Introducing the DJI Mavic 2-sampled.wmv')
