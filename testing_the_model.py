# test_model.py

import numpy as np
from Screenshot import screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random


WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Self-Driving-Car-{}-{}-{}.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

def Forward():

    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def Turn_left():
    PressKey(W)
    PressKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def Turn_Right():
    PressKey(D)
    PressKey(W)
    time.sleep(t_time)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,0,850,660))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (200,150))

            prediction = model.predict([screen.reshape(200,150,1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                forward()
            elif prediction[0] > turn_thresh:
                Turn_left()
            elif prediction[2] > turn_thresh:
                Turn_Right()

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                time.sleep(1)

main()       










