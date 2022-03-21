
import numpy as np
import cv2 as cv2
from mss import mss
from time import time
from PIL import Image, ImageEnhance, ImageOps, ImageGrab
#import keyboard 
import time
from vision import Vision

#import tqdm as tqdm
#import matplotlib.pyplot as plt
#import tensorflow as tf
#import random
'''
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
'''
from tqdm import tqdm

import win32gui, win32ui, win32con, win32api

#primeiro vamos captar a tela

def grab_screen(window_name = None, region=None):
    
    if window_name != None:
        hwin = win32gui.FindWindow(None, window_name)
    else:
        hwin = win32gui.GetDesktopWindow()
        if not hwin:
            raise Exception("Navegador nao encontrado")

    #print(win32gui.FindWindow(None, window_name))
    #hwin = win32gui.GetDesktopWindow()


    if region:
        left , top, x2, y2 = region
        width = x2 - left + 1
        height = y2 -top + 1
    else:
        
        #window_rect = win32gui.GetWindowRect(hwin)
        #width = window_rect[2] - window_rect[0]
        #height = window_rect[3] - window_rect[1]

        
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    ###########################
    #left,top,right,bot = win32gui.GetWindowRect(hwin)
    #width = right - left
    #height = bot - top
    ############################


    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc,width, height)

    memdc.SelectObject(bmp)

    memdc.BitBlt((0,0), (width, height), srcdc, (left,top), win32con.SRCCOPY)


    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

   

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


sct = mss()
mon = {'top': 0, 'left': 0, 'width': 1600, 'height': 1024}
loop_time = time.time()

def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

list_window_names()

#checar objeto
vision_doge = Vision('javali.png')

while True:
    #image = wincap.get_screenshot()
    
    image = grab_screen()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.Canny(image, threshold1=200, threshold2=300)
    #image = cv2.resize(image, (600,600))
    cv2.imshow("doge", image)


    #points= vision_doge.find(image,0.5,'rectangles')





    #print('FPS {}'.format(1 / (time.time() - loop_time)))
    loop_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
