from msilib import datasizemask
from turtle import st
import pandas as pd
import cv2,time


from cv2 import THRESH_BINARY
from datetime import datetime

first_frame=None
status_list=[None,None]
times=[]
df =pd.DataFrame(columns=["Start","End",])
video=cv2.VideoCapture(0) 
while True:
    check ,frame=video.read()
    status=0
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
        status=1
        (x,y,w,h)=cv2.boundingRect(contours)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)


    if status_list[-1]==1 and status_list[-2]==0 :
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1 :    
        times.append(datetime.now())




    cv2.imshow("grayframe",gray)
    cv2.imshow("deltaframe",delta_frame)
    cv2.imshow("Threshold frame",thresh_frame)
    cv2.imshow("Color frame",frame)

    key=cv2.waitKey(1)
    #keyword to quit the loop
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
for i in range(0,len(times),2):
   df = df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("times.csv")    

video.release()
cv2.destroyAllWindows()