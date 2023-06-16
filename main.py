# CoolCam Final Project
#? imports
import tkinter as tk
import cv2
import numpy as np

#! TODO: MAKE A BGR BUTTON CUZ ITS COOL, ALSO CHANGE THINGS ONLY IN THE ROI

#? Initialize variables
close_app = False

#* Effects
is_blacknwhite = False
is_invert = False
is_mask = False
is_facerec = False
is_eyerec = False
blinker = False
before = False
justblinked = False
blinked = False
is_umair = False

#* Mask Data
lower_red = np.array([30,150,50])
upper_red = np.array([255,255,255])
lower_green = np.array([50,100,100])
upper_green = np.array([100,255,255])
lower_blue = np.array([90,50,50])
upper_blue = np.array([130,255,255])

#* Face Detection HAAR Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
umair_cascade = cv2.CascadeClassifier('cascades/pc7cascade.xml') #* cascade that I trained

#? Initialize functions
def on_closing():
    global close_app
    close_app = True

def facerec_builtin():
    global is_facerec
    global is_eyerec
    if not is_facerec and not is_eyerec:
        is_facerec = True
        is_eyerec = True
        return
    elif is_facerec and is_eyerec:
        is_eyerec = False
        return
    elif is_facerec and not is_eyerec:
        is_facerec = False
        is_eyerec = True
        return
    elif not is_facerec and is_eyerec:
        is_facerec = False
        is_eyerec = False

def masker():
    global is_mask
    if is_mask == False:
        is_mask = 'red'
        return
    if is_mask == 'red':
        is_mask = 'green'
        return
    if is_mask == 'green':
        is_mask = 'blue'
        return
    if is_mask == 'blue':
        is_mask = False
        return
    print(is_mask)

def invert():
    global is_invert
    if is_invert == False:
        is_invert = True
    else:
        is_invert = False

def blacknwhite():
    global is_blacknwhite
    if is_blacknwhite == False:
        is_blacknwhite = True
    else:
        is_blacknwhite = False

def blinkaction():
    global blinker
    if blinker == False:
        blinker = 'invert'
        root.button_blink.configure(text='Blink Action:\nInvert')
        return
    elif blinker == 'invert':
        blinker = 'bw'
        root.button_blink.configure(text='Blink Action:\nB&W')
        return
    elif blinker == 'bw':
        blinker = False
        root.button_blink.configure(text='Blink Action:\nDisabled')
        return

def umairswitch():
    global is_umair
    if not is_umair:
        is_umair = True
        return
    if is_umair:
        is_umair = False
        return

#? TKINTER window setup
root = tk.Tk() #* create tkinter object
#* main window
root.title('')
root.protocol("WM_DELETE_WINDOW", on_closing)
root.resizable(True, False)

#* UI Buttons
icon_facerec = tk.PhotoImage(file= 'resources/facerec.png').subsample(8,8)
icon_mask = tk.PhotoImage(file= 'resources/colormask.png').subsample(11,11)
icon_invert = tk.PhotoImage(file= 'resources/invert.png').subsample(5,5)
icon_bw = tk.PhotoImage(file= 'resources/bw.png').subsample(2,2)
icon_umair = tk.PhotoImage(file= 'resources/umair.png').subsample(2,2)

#* side buttons
root.title = tk.Label(root, text='Settings', font=('Arial', 20), pady=5)
root.title.grid(row=0, column=0)
root.button_facerec = tk.Button(root, image= icon_facerec, command=facerec_builtin)
root.button_facerec.grid(row = 1, column = 0)
root.button_umairrec = tk.Button(root, image = icon_umair, command=umairswitch)
root.button_umairrec.grid(row = 2, column = 0)
root.button_mask = tk.Button(root, image = icon_mask, command=masker)
root.button_mask.grid(row = 3, column = 0)
root.button_invert = tk.Button(root, image= icon_invert, command=invert)
root.button_invert.grid(row = 4, column = 0)
root.button_blacknwhite = tk.Button(root, image= icon_bw, command=blacknwhite)
root.button_blacknwhite.grid(row = 5, column = 0)
root.button_blink = tk.Button(root, text= 'Blink Action:\nDisabled', command=blinkaction)
root.button_blink.grid(row = 6, column = 0)

#? LOOP and CV2 Code
vid = cv2.VideoCapture(0) #* create video capture object
width = int(vid.get(3))
height= int(vid.get(4))
while(True):
    root.update()
    ret, frame = vid.read() #* capture the video frame by frame

    #* ALL FILTERS AND EFFECTS HERE
    #* Masks
    if is_mask != False:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if is_mask == 'red':
            mask = cv2.inRange(hsv, lower_red,upper_red)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        if is_mask == 'green':
            mask = cv2.inRange(hsv, lower_green,upper_green)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        if is_mask == 'blue':
            mask = cv2.inRange(hsv, lower_blue,upper_blue)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    #* Face and Eye Recognition (built-in algorithm)
    if is_facerec and is_eyerec:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 127), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 191, 255), 3)

            #* Blink Detection (only works when eye detection enabled)
            if len(eyes) == 2:
                before = True
            if len(eyes) == 0 and before == True:
                justblinked = True
            if len(eyes) == 2 and justblinked == True:
                blinked = True

    elif is_facerec:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 127), 3)
    elif is_eyerec:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 191, 255), 3)
        
        #* Blink Detection (only works when eye detection enabled)
            if len(eyes) == 2:
                before = True
            if len(eyes) == 0 and before == True:
                justblinked = True
            if len(eyes) == 2 and justblinked == True:
                blinked = True
    #* My Trained AI
    if is_umair:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        umairs = umair_cascade.detectMultiScale(gray, 2, 5)
        for (x,y,w,h) in umairs:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(89,89,255),3)
    
    #* Blink Effects
    if blinked and blinker == 'invert':
        invert()
        before = False
        justblinked = False
        blinked = False
    if blinked and blinker == 'bw':
        blacknwhite()
        before = False
        justblinked = False
        blinked = False
    #* Color Filters
    if is_invert == True:
        frame = np.invert(frame)
    if is_blacknwhite == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.flip(frame, 1)
    cv2.imshow('CoolCam', frame) #* create cv2 window to display feed
    if cv2.waitKey(1) == ord('q'): #! q closes both when in focus of the cv window
        break
    if close_app:
        break
vid.release() #* after breaking the loop, release control of the camera
cv2.destroyAllWindows() #* destroy cv2 window