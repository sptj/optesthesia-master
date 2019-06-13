import cv2
import numpy

p=cv2.imread(r'H:\PycharmProjects\optesthesia\retinanet\root.png')

gray=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
ret,th1=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

cv2.imwrite('th1.png',th1)





