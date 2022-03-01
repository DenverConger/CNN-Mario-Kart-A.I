'''
Written by Denver Conger

github.com/DenverConger

conda activate SAI
pip install pydirectinput
pip install pui
pip install pillow
pip install opencv-python
pip install keyboard

ask Derek Muller if RunMario is not importing correctly

Remember to throw your first batch of laps into the test folder and name them on the test_classes.csv
'''

import cv2
import numpy as np
import pyautogui as pui
import keyboard
from PIL import ImageGrab
import time
import pydirectinput
import os,sys

# local imports
from RunMario import Mario # added by Derek Muller - mul19007@byui.edu

# keys that are used to turn left, turn right, accelerate forward, and use item
LEFT = 'left'
RIGHT = 'right'
FORWARD = 'left shift'
SPECIAL = 'z'

def screen(box):
    im = pui.screenshot(region=box)
    img_np = np.array(im) 
    # # Convert RGB to BGR 
    frame = img_np
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    return frame

img_indexs = [0,0,0]
def collect_input_data(frame,M):
    global img_indexs
    # frame = screen(screen_box)
    if keyboard.is_pressed(M._LEFT):
        # remember to add the rest of your path to the folders!
        isWritten = cv2.imwrite(f'training/1\Left{img_indexs[0]}.png', frame)
        print(f"Photo Number a")
        img_indexs[0] += 1
    elif keyboard.is_pressed(M._RIGHT):
        isWritten = cv2.imwrite(f'training/2\Right{img_indexs[2]}.png', frame)
        print(f"Photo Number d")
        img_indexs[2] += 1
    # if keyboard.is_pressed(M._FORWARD):
    else:
        isWritten = cv2.imwrite(f'training/0\Straight{img_indexs[1]}.png', frame)
        print(f"Photo Number w")
        img_indexs[1] += 1
    # if keyboard.is_pressed(M._SPECIAL):
    #     if frame == None:
    #         frame = screen(screen_box)
    #     # isWritten = cv2.imwrite(f'simple_kart_ai/training/training/0\Straight{i3}.png', frame)
    #     isWritten = cv2.imwrite(f'training/0\Straight{img_indexs[1]}.png', frame)
    #     print(f"Photo Number w")
    #     img_indexs[1] += 1

def show_capture(frame):
    # img = ImageGrab.grab(bbox=(top_x,top_y,img_h,img_w)) #x, y, w, h
    # img = ImageGrab.grab(box) #x, y, w, h
    # img_np = np.array(img)
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # print(frame[100][175])
        # These frame coordinates go [y][x] starting from the top left
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        return -1
    else:
        return 1

def avg_center(w):
    c = w.center
    region = (c.x,c.y,25,25)
    frame = screen(region)
    return np.mean(frame)


if __name__=='__main__':
    M = Mario()
    if 'test_start' in sys.argv:
        print("Testing emulator open start")
        time.sleep(1)
        M.automate_opening()
        exit()
    elif 'test_mario' in sys.argv:
        M.start()
        exit()
    else:

        # open Mario and go through menu
        M.automate_opening()
        # get location of window
        M.set_window(pui.getActiveWindow())

        nosave = False
        if 'nosave' in sys.argv:
            nosave = True
            print("nosave is set to true")

        time.sleep(3)
        # record player's play through
        i1 = 0
        i2 = 0
        i3 = 0
        print("Starting loop")
        while True:
            box = pui.getActiveWindow().box
            frame = screen(box)
            if not nosave:
                collect_input_data(frame,M)
            if show_capture(frame) == -1:
                break

print('end')
'''
Written by Denver Conger
Edited by Derek Muller

github.com/DenverConger
github.com/dmuller104
'''