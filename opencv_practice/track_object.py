import cv2


cv2.Tra
video_capture=cv2.VideoCapture(r'E:\video_save_燕郊\video20181111_171822.mp4')
status, frame = video_capture.read()
cv2.imshow("frame",frame)
key=cv2.waitKey(0)& 0xFF
if key==ord('s'):
    initBB=cv2.selectROI("frame",frame,fromCenter=False,showCrosshair=True)
    print(initBB)
