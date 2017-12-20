import numpy as np
import cv2
import time

frontal_face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_eye.xml")
#profile_faces_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_profileface.xml")

cap = cv2.VideoCapture(0)

count = 0
count_faces = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    start = time.time()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontal_faces = frontal_face_cascade.detectMultiScale(gray,1.04,3)
    for(x,y,w,h) in frontal_faces:
        count_faces += 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    end = time.time()

    print("time elapsed : " + str(end - start))
    print("count_faces = " + str(count_faces))
    print("count = " + str(count))
    #profile_faces = profile_faces_cascade.detectMultiScale(gray,1.3,5)
    #for(x,y,w,h) in profile_faces:
    #    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Eyes detection
        #roi_gray = gray[y:y+h, x:x+h]
        #roi_color = frame[y:y+h, x:x+h]
        #eyes = frontal_face_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    end = time.time()
    print("FPS : " + str(1/(end-start)))

    #name = "frame%d.jpg"%count
    #cv2.imwrite(name, frame)  # save frame as JPEG file

    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
