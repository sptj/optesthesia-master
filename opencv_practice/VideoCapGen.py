import cv2
from os.path import exists
class VideoCapture:
    def __init__(self,video_file_path):
        if not exists(video_file_path):
            raise FileExistsError('video file error or stream path is not correct')
        self.cap=cv2.VideoCapture(video_file_path)
        self.frame_total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count=0
    def __enter__(self):
        while True:
            status, frame = self.cap.read()
            self.frame_count=self.frame_count+1
            print('debug->count',self.frame_count/self.frame_total)
            if status:
                yield frame
            else:
                break
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('debug->exc_type',exc_type)
        print('debug->exc_val',exc_val)
        print('debug->exc_tb',exc_tb)
        self.cap.release()



if __name__ == '__main__':
    with VideoCapture(r'G:\DroneDetect\ChalNashu\document\test\video201871_12957.mp4') as v:
        for img in v:
            print(img.shape)
