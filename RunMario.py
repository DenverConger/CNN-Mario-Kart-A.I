import time
# from daemon import runner
from multiprocessing import Process
from threading import Thread
import os,sys
import pydirectinput
import pyautogui as pui
import numpy as np
import cv2

def press(key):
    print("Pressing: ", key)
    pydirectinput.press(key)
class Mario(Thread):
    def __init__(self):
        self.driver = MarioPlayer()
        Thread.__init__(self)
        self.starter_path = 'mupen_emulator_files\\PlayMario.bat'
        self.window = None
    
    def run(self):
        print('start mario')
        os.system(self.starter_path)
        print('end')
    

    def auto_menu(self):
        print("Auto menu inputs")
        press('enter')
        time.sleep(3)

        press('enter')
        time.sleep(3)

        press('s')
        time.sleep(2)
        for i in range(15):
            press('enter')
            time.sleep(0.5)

    def automate_opening(self):
        # M = Mario()
        # M.start()
        self.start()
        time.sleep(1)
        for i in range(100):
            if Mario.avg_center(pui.getActiveWindow()) < 245:
                time.sleep(0.2)
            else:
                print('starting')
                time.sleep(4)
                break
        # Wait for emulator to load game
        # time.sleep(8)    # Automatically go through menus
        self.auto_menu()

    def avg_center(w):
        c = w.center
        region = (c.x,c.y,25,25)
        frame = Mario.screen(region)
        return np.mean(frame)

    def screen(box):
        # top_x,top_y,bottom_w,bottom_h,img_h,img_w = box
        im = pui.screenshot(region=box)
        # img = pui.screenshot()
        img_np = np.array(im) 
        # # Convert RGB to BGR 
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        frame = img_np
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # (thresh, frame) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        return frame
    
    def set_window(self,window):
        self.window = window



class MarioPlayer:
    def __init__(self):
        self._LEFT = 'left'
        self._RIGHT = 'right'
        self._FORWARD = 'shiftleft'
        self.downKeys = set()
    def keyDown(self,key):
        pui.keyDown(key)
        self.downKeys.add(key)
    def keyUp(self,key):
        pui.keyUp(key)
        if key in self.downKeys:
            self.downKeys.remove(key)
    def keyUpAll(self):
        for key in self.downKeys:
            self.keyUp(key)
    def key_command(self,keys,time_down=1,keepDown=False):
        for key in keys:
            self.keyDown(key)
        time.sleep(time_down)
        if not keepDown:
            for key in keys:
                self.keyUp(key)
    def move(self,direction,cmdTime=1,keepDown=False):
        if direction == 'left':
            print('move left')
            self.key_command([self._LEFT,self._FORWARD],cmdTime,keepDown)
        elif direction == 'right':
            print('move right')
            self.key_command([self._RIGHT,self._FORWARD],cmdTime,keepDown)
        elif direction == 'straight':
            print('move straight')
            self.key_command([self._FORWARD],cmdTime,keepDown)
                
    def forward_left(self,cmdTime=1):
        self.key_command((self._LEFT,self._FORWARD),cmdTime)    


def play_through():
    M = Mario()
    M.automate_opening()
    time.sleep(6)
    M.driver.move('straight',11)
    for _ in range(14):
        M.driver.move('left',0.005)
        M.driver.move('straight',0.4)



if __name__ == '__main__':
    try:
        M = Mario()
        if 'menu' in sys.argv:
            M.automate_opening()
        elif 'move_forward' in sys.argv:
            M.automate_opening()
            MP = MarioPlayer()
            time.sleep(10)
            MP.move('straight',9)
            MP.move('left',3)
            MP.move('straight',4)
            MP.move('right',2)
        elif 'move_forward2' in sys.argv:
            M.automate_opening()
            MP = MarioPlayer()
            time.sleep(10)
            for _ in range(10):
                MP.move('straight',12)
                MP.move('left',1)
                # MP.move('right',0.25)
            # MP.move('straight',9)
            # MP.move('left',3)
            # MP.move('straight',4)
            # MP.move('right',2)
        elif 'play_through' in sys.argv:
            play_through()
        else:
            M = Mario()
            M.automate_opening()
            # M.start()
        # M.start()
        # mario_process = Process(target=M.run,daemon=True)
        # mario_process = Process(target=M.run)
        # mario_process.start()
        time.sleep(10)
        print('ending')
        # M.run()
    finally:
        MP.keyUpAll()