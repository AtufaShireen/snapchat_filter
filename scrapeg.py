import cv2
import numpy as np

eyes_path = './Train/third-party/frontalEyes35x16.xml'
nose_path = './Train/third-party/Nose18x15.xml'
eye_casscade = cv2.CascadeClassifier(eyes_path)
nose_casscade = cv2.CascadeClassifier(nose_path)

face_casscade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

glasses = cv2.imread("Train/glasses4.png", cv2.IMREAD_UNCHANGED)
print(glasses.shape)
moustache = cv2.imread("Train/mustache.png", cv2.IMREAD_UNCHANGED)

capture = 0
while True:

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        print('Erorrr')
        continue
    faces = face_casscade.detectMultiScale(frame, 1.3, 5)
    eyes = eye_casscade.detectMultiScale(frame, 1.1, 5)
    noses = nose_casscade.detectMultiScale(frame, 1.1, 3)

    for nose in noses:
        x, y, w, h = nose
        # cv2.rectangle(man, (x, y), (x + w, y + h), (0, 0, 250), 2)
        moustache = cv2.resize(moustache, (w, h))
        for i in range(moustache.shape[0]):
            for j in range(moustache.shape[1]):
                if moustache[i, j, 3] > 0:
                    frame[y + i, x + j, :] = moustache[i, j, :-1]

    for eye in eyes:
        x, y, w, h = eye
        # cv2.rectangle(man, (x, y), (x + w, y + h), (0, 0, 250), 2)
        glasses = cv2.resize(glasses, (w, h))
        for i in range(glasses.shape[0]):
            for j in range(glasses.shape[1]):
                if glasses[i, j, 3] > 0:
                    frame[y + i, x + j, :] = glasses[i, j, :-1]


    cv2.imshow('YOur Camera', frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
