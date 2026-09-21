import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


# =========================
# Configuration
# =========================

CAMERA_INDEX = 0
IMG_SIZE = 300
OFFSET = 20

MODEL_PATH = r"C:\Users\Titif\OneDrive\Documents\Sign game\model\keras_model.h5"
LABELS_PATH = r"C:\Users\Titif\OneDrive\Documents\Sign game\model\labels.txt"

LABELS = ["1", "2", "3", "4", "5", "10"]


# =========================
# Camera
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================
# Hand detector
# =========================

detector = HandDetector(maxHands=1)


# =========================
# Classifier
# =========================

classifier = Classifier(
    MODEL_PATH,
    LABELS_PATH
)


# =========================
# Variables
# =========================

prediction = ""
index = 0


# =========================
# Main loop
# =========================

while True:

    success, img = cap.read()

    if not success:
        print("Failed to read camera.")
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(
        img,
        draw=False
    )

    if hands:

        hand = hands[0]

        x, y, w, h = hand["bbox"]

        # Make sure the crop stays inside the image
        y1 = max(0, y - OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)

        x1 = max(0, x - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        # White image for the classifier
        imgWhite = np.ones(
            (IMG_SIZE, IMG_SIZE, 3),
            np.uint8
        ) * 255

        aspectRatio = h / w

        # =========================
        # Vertical hand
        # =========================

        if aspectRatio > 1:

            k = IMG_SIZE / h

            wCal = math.ceil(k * w)

            if wCal > 0:

                imgResize = cv2.resize(
                    imgCrop,
                    (wCal, IMG_SIZE)
                )

                wGap = math.ceil(
                    (IMG_SIZE - wCal) / 2
                )

                imgWhite[
                    :,
                    wGap:wGap + wCal
                ] = imgResize

        # =========================
        # Horizontal hand
        # =========================

        else:

            k = IMG_SIZE / w

            hCal = math.ceil(k * h)

            if hCal > 0:

                imgResize = cv2.resize(
                    imgCrop,
                    (IMG_SIZE, hCal)
                )

                hGap = math.ceil(
                    (IMG_SIZE - hCal) / 2
                )

                imgWhite[
                    hGap:hGap + hCal,
                    :
                ] = imgResize

        # =========================
        # Prediction
        # =========================

        prediction, index = classifier.getPrediction(
            imgWhite,
            draw=False
        )

        print(prediction, index)

        # Make sure index is valid
        if 0 <= index < len(LABELS):

            label = LABELS[index]

        else:

            label = "Unknown"

        # =========================
        # Display prediction
        # =========================

        cv2.putText(
            imgOutput,
            label,
            (x, max(50, y - 20)),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 0, 255),
            3
        )

        # =========================
        # Hand rectangle
        # =========================

        cv2.rectangle(
            imgOutput,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            4
        )

        # =========================
        # Display windows
        # =========================

        """cv2.imshow(
            "ImgCrop",
            imgCrop
        )"""

        """cv2.imshow(
            "ImgWhite",
            imgWhite
        )"""

    cv2.imshow(
        "Image",
        imgOutput
    )

    # =========================
    # Keyboard controls
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================
# Release
# =========================

cap.release()
cv2.destroyAllWindows()