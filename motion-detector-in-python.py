import cv2,time

from cv2 import THRESH_BINARY

first_frame=None

video=cv2.VideoCapture(0) 
while True:
    check ,frame=video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    #finding first frame
    if first_frame is None:
        first_frame=gray
        continue
    
    #finding difference between the fisrt image and  the new image
    delta_frame=cv2.absdiff(first_frame,gray)
    #defining threshold to turn area into black or white according to observed motion
    thresh_delta=cv2.threshold(delta_frame,30,255,THRESH_BINARY)[1]
    #dilating the white area
    thresh_frame=cv2.dilate(thresh_delta,None,iterations=2)
    #finding contours to detect motion
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contours in cnts:
        if cv2.contourArea(contours) < 1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contours)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("grayframe",gray)
    cv2.imshow("deltaframe",delta_frame)
    cv2.imshow("Threshold frame",thresh_frame)
    cv2.imshow("Color frame",frame)

    key=cv2.waitKey(1)
    #keyword to quit the loop
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()