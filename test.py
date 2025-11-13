import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Settings
offset = 20
imgSize = 300

# Load labels (A–Z and 0–9)
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand image
        y1 = max(0, y - offset)
        y2 = min(y + h + offset, img.shape[0])
        x1 = max(0, x - offset)
        x2 = min(x + w + offset, img.shape[1])

        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        # Resize and center the image
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = labels[index]
        print(f"Prediction: {label} | Confidence: {np.max(prediction):.2f}")

        # Draw on output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Show the cropped and white background images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
