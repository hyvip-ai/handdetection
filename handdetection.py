import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

while True:
    suc,img = cap.read()
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

    cv2.imshow('OUTPUT',img)
    cv2.waitKey(1)
