# Import OpenCV and MediaPipe
import cv2
import mediapipe as mp

import numpy as np
import uuid
import os

from tensorflow.keras.models import load_model

labels=['rock','paper','scissors']

# # Draw hands
mp_drawing=mp.solutions.drawing_utils # Drawing utilities
mp_hands=mp.solutions.hands # Model of hand with landmarks and lines

cap=cv2.VideoCapture(0)

# Load gesture recognizer model
# model=load_model('thisbetterwork.h5') # We are going to make our own model

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret,frame=cap.read()
        x,y,c=frame.shape
        
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        roi=frame[0:225,0:225]
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        roi.flags.writeable=False # What does this do ??? 
        results=hands.process(roi) #Detections
        roi.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        roi=cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        # print(results.multi_hand_landmarks) # Tells position of landmarks in x,y,z coordinates

        # Rendering results
        if results.multi_hand_landmarks:
            landmarks=[]
            for handsLms in results.multi_hand_landmarks:  
                for lm in handsLms.landmark:
                    lmx=int(lm.x*x)
                    lmy=int(lm.y*y)
                    landmarks.append([lmx,lmy])
                mp_drawing.draw_landmarks(roi,handsLms,mp_hands.HAND_CONNECTIONS)

                # Predict gesture
                # prediction=model.predict([landmarks])
                # classID=np.argmax(prediction)
                # class_name=labels[classID].capitalize()

        # cv2.putText(frame,class_name,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)     
        #Show image
        cv2.imshow('Hand Tracking', image)
        cv2.imshow('bruh', roi)

        if cv2.waitKey(10) & 0xff==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Output images