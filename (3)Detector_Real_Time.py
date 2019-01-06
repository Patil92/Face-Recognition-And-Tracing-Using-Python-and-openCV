import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainningData.yml")
id=0
ch='Not in Database..!!'
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,2)
while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                if(id==1):
                        ch="Prajwal G Kadakol"
                        cv2.cv.PutText(cv2.cv.fromarray(img),str('SR NO. : 160036'),(x,y+h+20),font,255);
                elif(id==2):
                        ch="Pranitha"
                        cv2.cv.PutText(cv2.cv.fromarray(img),str('SR NO. : 163721'),(x,y+h+20),font,255);
                elif(id==3):
                        ch="Y S D Sir"
                        cv2.cv.PutText(cv2.cv.fromarray(img),str('SR NO. : 163721'),(x,y+h+20),font,255);
                else:
                        ch='Not in Database..!!'

                cv2.cv.PutText(cv2.cv.fromarray(img),str(ch),(x,y+h),font,255);
                #cv2.cv.PutText(cv2.cv.fromarray(img),str('B Sec ECE Dept.'),(x,y+h+40),font,255);
        
        cv2.imshow("Face",img);
        if(cv2.waitKey(1)==ord('q')):
                break;
cam.release()
cv2.destroyAllWindows()
