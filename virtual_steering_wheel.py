import cv2 as cv
import mediapipe as mp
import time
import math
from directinput import release_key, press_key

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplex = 1,  detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,  self.detectionCon, self.trackCon )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)    # Converting a BGR image to RGB, to pass the image through the mediapipe process
        self.result = self.hands.process(imgRGB)        # Passing the RGB image to hands.process function, this will return all the data of hands
        
        landmarks = self.result.multi_hand_landmarks    # the the landmarks (cordinates) of each 21 landmarks of each hand as a List
        
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)     # Drawing the landmarks and connections
                        
        if landmarks:
            if len(landmarks) == 2:      # Checking if there are two hands
                    
                left_hand_landmarks = landmarks[1].landmark   # Left Hand Data
                right_hand_landmarks = landmarks[0].landmark  # right Hand Data

                shape = img.shape   # Extrating the video resolution 
                width = shape[1]
                height = shape[0]

                left_iFingerX, left_iFingerY = (left_hand_landmarks[5].x * width), (left_hand_landmarks[5].y * height)      # Converting the landmarks to pixelated values (cordinates(x, y))
                right_iFingerX, right_iFingerY = (right_hand_landmarks[5].x * width), (right_hand_landmarks[5].y * height)  # Same for the right hand

                slope = ((right_iFingerY - left_iFingerY)/(right_iFingerX-left_iFingerX))       # Calculating the slope 

                # hy = math.hypot(abs(right_iFingerX-left_iFingerX), abs(right_iFingerY - left_iFingerY))     # Calculating the distance between the hands to use in future
                
                cv.line(img,(int(left_iFingerX), int(left_iFingerY) - 40), (int(right_iFingerX), int(right_iFingerY) - 40), (0,255,255), thickness = 3)       # Drawing the line between the hands
                cv.circle(img, ((int(left_iFingerX) + int(right_iFingerX))//2, ((int(left_iFingerY) + int(right_iFingerY) - 70))//2), 40, (0,255,0), thickness = 2)
                

                # Print statements for Debugging

                # print(f"Slope: {slope}")
                # print(hy)

                sensitivity = 0.25      # senstivity of the turn 
                if abs(slope) > sensitivity:
                    if slope < 0:           # Turning left
                        release_key("w")
                        release_key('a')
                        press_key('a')
                    if slope > 0:           # Turning right
                        release_key('w')
                        release_key('a')
                        press_key('d')
                if abs(slope) < sensitivity:    # Accelerate while still
                    release_key('a')
                    release_key('d')
                    press_key('w')

        return img
    
    def findPosition(self, img, handNo = 0, draw = True):       # Returning the list of all the 21 points to use in future
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lmList


def main():

    pTime = 0   
    cTime = 0
    
    cap = cv.VideoCapture(0)
    # cap.set(3, 1280)      # Tweak, if want to change the size of camera window 
    # cap.set(4, 720)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        cv.waitKey(1)
        img = detector.findHands(img)

        ####################################################################################
                    # For FPS counter
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img,'FPS : ' +  str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        ####################################################################################

        cv.imshow('Image', img)         # Displaying the video

if __name__ == "__main__":
    main()
