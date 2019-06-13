import cv2
from managers import WindowManager,CaptureManage
class Cameo(object):
    def __init__(self,filepath):
        self._windowManager=WindowManager('Cameo',self.onKeypress)
        video_capture=cv2.VideoCapture(filepath)
        print(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,640*4)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240*4)
        print(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._captureManager=CaptureManage(video_capture,self._windowManager,True)
    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame=self._captureManager.frame
            # write your code here


            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    def onKeypress(self,keycode):
        if keycode == ord(' '):
            self._captureManager.writeImage('screenshot.png')
        elif keycode ==ord('r'):
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode==27:#escape
            self._windowManager.destoryWindow()


if __name__ == '__main__':
    Cameo(r'K:\video2019530_143844.mp4').run()















