import cv2
from datetime import datetime
import time
import os
class video_recoder(object):
    def __init__(self,width=None,height=None):
        def get_w_h():
            w=video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h=video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return w,h

        video_capture=cv2.VideoCapture(0)

        w, h = get_w_h()
        if width!=None and height!=None:
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            print('width:',w,'\t' ,'height:',h)
            w, h = get_w_h()
            if(w==width)and(h==height):
                print('config accepted')
            else:
                print('config refused')

        self.video_capture=video_capture
        self.width=w
        self.height=h
        self.fps=self.fps_detect()
        print(self.fps)
    def fps_detect(self):
        start_time=time.time()
        status, frame=self.video_capture.read()
        i=0
        while status:
            i=i+1
            status, frame = self.video_capture.read()
            end_time=time.time()
            if(end_time-start_time)>5:
                return int(i / (end_time - start_time))
        return 0
    def execute(self):
        now_time=datetime.now()
        video_name=now_time.strftime("%Y-%m-%d-%H-%M-%S")
        video_file_name="{}.avi".format(video_name)
        fourcc_specific={
            '.mp4':'MJPG',
            '.avi':'XVID',
            '.ogv':'THEO',
            '.flv':'FLV1',
            '.wmv':'MJPG',
            '.mkv':'3IVX',
        }
        (_, file_ext) = os.path.splitext(video_file_name)
        file_ext = file_ext.lower()
        if file_ext not in fourcc_specific:
            print('dist format not support')
            exit(0)
        fourcc_type=fourcc_specific[file_ext]
        video_writer = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*fourcc_type), self.fps, (int(self.width),int(self.height)))
        status,frame=self.video_capture.read()
        i = 0
        while status:
            i = i + 1
            video_writer.write(frame)
            status, frame = self.video_capture.read()

if __name__ == '__main__':
    video_recoder(2560,960).execute()
