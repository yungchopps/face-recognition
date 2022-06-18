import cv2
import numpy as np


def main():
    print('OpenCV version:' + cv2.__version__)

    # Initialize the face detection classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # PC webcam is video source
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Webcam was not open correctly")

    while True:
        # Reads the frame
        ret, frame = cap.read()
        # Grays the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects the face
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        cv2.imshow('Gray', gray)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()