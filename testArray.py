import os

import cv2
import face_recognition
import numpy as np
from numpy import asarray

camera = cv2.VideoCapture(1)  # use 0 for web camera
# SITE_ROOT = os.path.dirname(os.path.realpath(__file__))

#
# known_face = "ImageDataset/ilham.jpeg"
found_face = "ImageFound/capture.jpg"
# found_face = "ImageFound/unknown.jpg"

images = []
newImages = []
classNames = []
foundNames = []

path = 'ImageDataSet'
newPath = 'ImageFound'
myList = os.listdir(path)
newList = os.listdir(newPath)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

for nl in newList:
    newImg = cv2.imread(f'{newPath}/{nl}')
    newImages.append(newImg)
    foundNames.append(os.path.splitext(nl)[0])
print(foundNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        # db.child("encodeList").push(encodeList)
    return encodeList


def findNewEncodings(newImages):
    encodeNewList = []
    for imgN in newImages:
        imgN = cv2.cvtColor(imgN, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgN)[0]
        encodeNewList.append(encode)
        # db.child("encodeList").push(encodeList)
    return encodeNewList


encodeListKnown = findEncodings(images)
encodeNewKnown = findNewEncodings(newImages)

print(encodeListKnown, '\n ', encodeNewKnown)

imgFound = cv2.imread(found_face)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rgbImgS = cv2.cvtColor(imgFound, cv2.COLOR_BGR2RGB)
facesCurFrame = face_recognition.face_locations(rgbImgS)
encodesCurFrame = face_recognition.face_encodings(rgbImgS, facesCurFrame)

for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.5)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

print(matches, faceDis)
