import numpy as np
import cv2
import os


def random_image_gen_by_np(shape):
    ri = np.random.normal(size=shape, loc=125, scale=125)
    return ri.astype(np.uint8)


def random_image_gen_by_os(shape):
    num = 1
    for i in shape:
        num = num * i
    ri = os.urandom(num)
    ri = np.array(bytearray(ri)).reshape(shape)
    return ri.astype(np.uint8)


def cvt_gray_to_bgr(gray_image):
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def set_item_by_different_routine(image, value, pos, routine_options='simple'):
    def simple_routine(image, value, pos):
        image[0][0] = value
        return image

    def complex_routine(image, value, pos):
        image.itemset(pos, value)
        return image

    return simple_routine(image, value, pos)


def cvt_image_to_bytearray(image, debug='off'):
    bytearray_image = bytearray(image)
    if debug == "on":
        print(bytearray_image)
    return bytearray_image


def cvt_image_to_bytes(image, debug='off'):
    bytes_image = bytes(image)
    if debug == "on":
        print(bytes_image)
    return bytes_image


def get_image_from_bytes(bytes_image, size, debug='off'):
    image = np.array(bytearray(bytes_image)).reshape(*size)
    if debug == "on":
        print(image)
    return image


def video_transfer(src_filename, dst_filename):
    if not os.path.exists(src_filename):
        print('source file not found')
        exit(0)
    video_capture = cv2.VideoCapture(src_filename)
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
    video_writer = cv2.VideoWriter(dst_filename, cv2.VideoWriter_fourcc(*fourcc_type), fps, size)

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


def video_transfer_test():
    video_transfer(r'video2018620_9555.mp4',
                   r'video2018620_9555.wmv')


def show_image(image):
    cv2.imshow('_', image)
    cv2.waitKey()


def play_image(image):
    cv2.imshow('_', image)
    cv2.waitKey(2)


def video_2_images(src_filename):
    if not os.path.exists(src_filename):
        print('source file not found')
        exit(0)
    video_capture = cv2.VideoCapture(src_filename)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    (file_path, file_name) = os.path.split(src_filename)
    (file_bald_name, file_ext_name) = os.path.splitext(file_name)
    file_path = os.path.join(file_path, file_bald_name)
    if os.path.exists(file_path):
        pass
    else:
        os.mkdir(file_path)
    status, frame = video_capture.read()
    i = 0
    while status:
        i = i + 1
        status, frame = video_capture.read()
        image_name = "{}_{:08}.png".format(file_bald_name, i)
        image_full_path = os.path.join(file_path, image_name)
        result = cv2.imwrite(image_full_path, frame)
        if result != True:
            print(image_full_path)
            print("error occur")
        if (i % 100 == 0):
            print('processed', int(i / frame_count * 100), '%')
    video_capture.release()


if __name__ == '__main__':
    video_2_images(r"M:\高淳视频\video201961_165710.mp4")
