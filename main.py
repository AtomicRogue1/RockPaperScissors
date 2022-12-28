# Import OpenCV and MediaPipe
import cv2
import mediapipe as mp

import numpy as np
import uuid
import os

# # Draw hands
mp_drawing=mp.solutions.drawing_utils # Drawing utilities
mp_hands=mp.solutions.hands # Model of hand with landmarks and lines

cap=cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret,frame=cap.read()

        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False # What does this do ??? 
        results=hands.process(image) #Detections
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # print(results.multi_hand_landmarks) # Tells position of landmarks in x,y,z coordinates

        # Rendering results
        if results.multi_hand_landmarks:z
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xff==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Output images