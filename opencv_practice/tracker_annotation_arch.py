import os
import sys

import cv2
from pascal_voc_io import PascalVocWriter

video_path = r"J:\高淳视频\video201961_165710.mp4"
output_dir = r"J:\gaochun\video201961_165710"
# output_dir=output_dir.encode('').decode()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def MouseEventCallBack(event, x, y, flags, param):
    print(event)
    if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_RBUTTONDBLCLK or event == cv2.EVENT_RBUTTONUP:
        print('called')
        global track_state_is_normal
        track_state_is_normal = False


if __name__ == '__main__':
    track_state_is_normal = True
    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = {'BOOSTING': cv2.TrackerBoosting_create,
                     'MIL': cv2.TrackerMIL_create,
                     'KCF': cv2.TrackerKCF_create,
                     'TLD': cv2.TrackerTLD_create,
                     'MEDIANFLOW': cv2.TrackerMedianFlow_create,
                     'GOTURN': cv2.TrackerGOTURN_create,
                     'CSRT': cv2.TrackerCSRT_create
                     }
    tracker_create = tracker_types['CSRT']
    tracker = tracker_create()
    # Read video
    video = cv2.VideoCapture(video_path)
    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()
    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    cv2.namedWindow("Tracking")
    bbox = cv2.selectROI("Tracking", frame, False)
    cv2.setMouseCallback("Tracking", MouseEventCallBack)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    i = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        i = i + 1
        if not ok:
            break
        image = frame.copy()
        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # Draw bounding box
        if (ok and track_state_is_normal):
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            tracker = tracker_create()
            bbox = cv2.selectROI("Tracking", frame, False)
            try:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                # cv2.imshow('test',frame)
                cv2.setMouseCallback("Tracking", MouseEventCallBack)
                track_state_is_normal = True
                # Initialize tracker with first frame and bounding box
                ok = tracker.init(frame, bbox)
            except:
                continue
        # imageShape = [frame.height, frame.width, 3]
        imageShape = image.shape
        imgFolderName = output_dir
        imgFileName = os.path.split(video_path)[-1].split('.')[0] + '_frame_' + str(i) + '.png'
        writer = PascalVocWriter(imgFolderName, imgFileName, imageShape)
        writer.verified = False
        # try:
        writer.addBndBox(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), '1', 0)
        targetPath = '{}_frame_{:08}'.format(os.path.join(output_dir, os.path.split(video_path)[-1].split('.')[0]), i)
        targetFile = targetPath + '.xml'
        targetImage = targetPath + '.png'
        writer.save(targetFile)
        result=cv2.imwrite(targetImage, image)
        print(targetImage,result)
        # Display tracker type on frame
        cv2.putText(frame, " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == 'q':
            print("esc pressed, all the program quit")
            break
