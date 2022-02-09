'''
written by Denver Conger

conda activate SAI
pip install pydirectinput
pip install pyautogui
pip install pillow
'''
import os
import tensorflow as tf
import pyautogui
import cv2
import numpy as np
import keyboard
from PIL import ImageGrab
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import time
import pydirectinput


model = tf.keras.models.load_model('kart_model')


def locate():
    # add 2 screenshots of the upper right and upper left corners to a folder
    emulatortop = pyautogui.locateOnScreen('Notepad.png')
    emulatorbottom = pyautogui.locateOnScreen('Notepad_2.png')
    print(emulatortop)
    print(emulatorbottom)\
    # Pyautogui wants the coordinates of top left plus the width
    top_x = emulatortop[0]
    top_y = emulatortop[1]
    bottom_h = emulatorbottom[1]+emulatorbottom[3]-emulatortop[1]
    # print(bottom_b)
    bottom_w = emulatorbottom[0]+emulatorbottom[2]-emulatortop[0]

    # Pillow though wants the actual coordinates of all 4 corners so we need 2 arrays of pixels
    img_h = emulatorbottom[1]+emulatorbottom[3]
    img_w = emulatorbottom[0]+emulatorbottom[2]
    locate_data = top_x+200,top_y+200,bottom_w-400,bottom_h-350,img_h-350,img_w-400 
    return locate_data



def screen(locate_data):
    top_x,top_y,bottom_w,bottom_h,img_h,img_w = locate_data
    im = pyautogui.screenshot(region=(top_x,top_y,bottom_w,bottom_h))


    img_np = np.array(im) 

    frame = img_np

    
    print(frame.shape)

    frame = cv2.resize(frame, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)

    frame = frame.reshape(-1, 200, 200, 3)

    


    return frame

def automate_opening():
    # REMEMBER TO ADD YOUR FULL PATH
    os.startfile("mupen64plus/mupen64plus/mupen64plus-gui.exe")
    time.sleep(2)

    button = pyautogui.locateOnScreen('file.png')
    button7point = pyautogui.center(button)
    button7x, button7y = button7point
    pyautogui.click(button7x, button7y)

    button = pyautogui.locateOnScreen('rom.png')
    button7point = pyautogui.center(button)
    button7x, button7y = button7point
    pyautogui.click(button7x, button7y)
    time.sleep(2)

    button = pyautogui.locateOnScreen('kart.png')
    button7point = pyautogui.center(button)
    button7x, button7y = button7point
    pyautogui.click(button7x, button7y)


    pydirectinput.press('enter')
    time.sleep(3)
    # This locate on screen should be a screenshot of just the grayed out emulator screen so it clicks on it
    button = pyautogui.locateOnScreen('Notepad.png')
    button7point = pyautogui.center(button)

    button7x, button7y = button7point
    pyautogui.click(button7x + 40, button7y + 40)
    pyautogui.click(button7x + 40, button7y + 40)
    pyautogui.click(button7x + 40, button7y + 40)
    # We only want to locate the dimentions once

    pydirectinput.press('enter')

    time.sleep(3)

    pydirectinput.press('enter')
    time.sleep(3)

    print("Enter Start 1")
    pydirectinput.press('enter')
    print("Start 1 pressed")
    time.sleep(3)

    print("Enter Start 2")
    pydirectinput.press('enter')
    print("Start 2 pressed")
    time.sleep(3)

    print("s pressing")
    pydirectinput.press('s')
    print("s pressed")
    time.sleep(2)

    pydirectinput.press('enter')
    time.sleep(1)
    pydirectinput.press('enter')
    time.sleep(1)
    pydirectinput.press('enter')
    time.sleep(3)

    pydirectinput.press('enter')
    time.sleep(1)
    pydirectinput.press('enter')
    time.sleep(3)

    pydirectinput.press('enter')
    time.sleep(1)
    pydirectinput.press('enter')
    time.sleep(1)

    pydirectinput.press('enter')
    time.sleep(1)


while True:
    # we want to screen capture though every loop
    frame = screen(locate_data)

    # The prediction comes out as an array with 3 outputted variables of probrability [.4235152,.1021512,.57324354987]
    #                                                                              [0] or straight [1] or left [2] or right
    pred = model.predict(frame, verbose=1)

    # This is the list from above
    cl = []
    print(pred)
    # from zero to the length of the array or 0 to 2 (3 digits) I want you to tell me which of the three had the highest probrability
    for i in range(0, len(pred)):
        # turns it form this [.4235152,.1021512,.57324354987] to this [0,0,1]
        cl.append(np.argmax(pred[i]))
    # we turn the single variable into an array to see which one still has a value and return the location so [2]
    cl = np.array(cl)
    print(cl[0])

    # keyboard.press("w")
    if cl[0] == 0:
        keyboard.release("a")
        keyboard.release("d")
        keyboard.press("w")
    if cl[0] == 1:
        keyboard.press("w")

        keyboard.release("d")
        keyboard.press("a")
    if cl[0] == 2:
        keyboard.press("w")
        keyboard.release("a")
        keyboard.press("d")
    locate_data = locate()


    top_x,top_y,bottom_w,bottom_h,img_h,img_w = locate_data



    img = ImageGrab.grab(bbox=(top_x,top_y,img_h,img_w)) #x, y, w, h
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # print(frame[100][175])
        # These frame coordinates go [y][x] starting from the top left
    cv2.imshow("frame", img_np)

    if cv2.waitKey(1) & 0Xff == ord('q'):
        break