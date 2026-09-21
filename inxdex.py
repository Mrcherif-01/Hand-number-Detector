import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset=20
counter=0
imgSize=300
folder="data/1"
while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255


        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        

        aspectRatio= h/w
        if aspectRatio > 1 :
            k= imgSize/h
            w_cal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(w_cal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-w_cal)/2)
            imgWhite[:, wGap:w_cal+wGap] = imgResize
        
        else :
            k= imgSize/w
            h_cal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(imgSize,h_cal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-h_cal)/2)
            imgWhite[hGap:h_cal+hGap,:] = imgResize
            

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)