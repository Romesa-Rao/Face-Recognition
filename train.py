

import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()

window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
window.geometry('1200x580')
window.configure(background='White')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message = tk.Label(window, text=" National College of Business Administration & Economics" ,fg="white", bg="#1B7A1D" ,width=50  ,height=2,font=('Garamond', 20, 'bold'))
message.place(x=50, y=10)

########################################
lbl = tk.Label(window, text="Enter ID  :     ",bg="white", fg="#70158F"  ,font=('arial', 15, ' bold ') )
lbl.place(x=100, y=100)

txt = tk.Entry(window,width=20, bg="#ede9e8", fg="black",font=('Courier New', 15))
txt.place(x=230, y=100)

########################################
lbl2 = tk.Label(window, text="Enter Name :   ",bg="white", fg="#70158F"    ,height=2 ,font=('arial', 15, ' bold '))
lbl2.place(x=100, y=150)

txt2 = tk.Entry(window,width=20  ,bg="#ede9e8", fg="black",font=('Courier New', 15)  )
txt2.place(x=230, y=165)

########################################
lbl3 = tk.Label(window, text="Notification : ",bg="white", fg="#70158F"  ,font=('arial', 15, ' bold ') )
lbl3.place(x=100, y=230)

message = tk.Label(window, text="" ,width=30  ,bg="#ede9e8", fg="black", activebackground="Red",height=2, font=('Courier New', 15))
message.place(x=270, y=230)

########################################
lbl3 = tk.Label(window, text="Attendance : ",bg="white", fg="#70158F"  ,font=('arial', 15, ' bold underline'))
lbl3.place(x=100, y=310)

message2 = tk.Label(window, text="" ,fg="black" ,bg="#D39CE2",activeforeground = "green",width=30  ,height=2  ,font=('Courier New', 15, ' bold '))
message2.place(x=270, y=310)
########################################
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        

                sampleNum=sampleNum+1

                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('frame',img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                my_date = str(now.date())
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,my_date,dtString]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    fileName = "Attendance/attendance.csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()

    res=attendance
    message2.configure(text= res)

  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="white"  ,bg="#1B7A1D"  ,width=10,font=('Garamond', 13, 'bold'))
clearButton.place(x=500, y=100)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="white"  ,bg="#1B7A1D"  ,width=10  ,font=('Garamond', 13, 'bold'))
clearButton2.place(x=500, y=160)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="white"  ,bg="#70158F"  ,width=15  ,height=1,font=('Garamond', 15, ' bold '))
takeImg.place(x=100, y=450)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="white"  ,bg="#70158F"  ,width=15  ,height=1 ,font=('Garamond', 15, ' bold '))
trainImg.place(x=350, y=450)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="white"  ,bg="#70158F"  ,width=15  ,height=1 ,font=('Garamond', 15, ' bold '))
trackImg.place(x=600, y=450)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="Black"  ,bg="#DEB887"  ,width=15  ,height=2 ,font=('Garamond', 15, ' bold '))
quitWindow.place(x=850, y=450)
 
window.mainloop()