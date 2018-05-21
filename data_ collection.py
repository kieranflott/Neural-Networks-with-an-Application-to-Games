import numpy as np
from Screenshot import screenshot
import cv2
import time
from getkeys import key_check
import os

Forward =[1,0,0,0,0]
Slow =   [0,1,0,0,0]
Left =   [0,0,1,0,0]
Right =  [0,0,0,1,0]
No =    [0,0,0,0,1]

starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File alredy exists',starting_value)
        starting_value += 1
    else:
        print('No File, starting fresh!',starting_value)
        
        break


def output_keys(keys):
    output = [0,0,0,0,0]
    if 'W' in keys:
        output = Forward
    elif 'S' in keys:
        output = Slow
    elif 'A' in keys:
        output = Left
    elif 'D' in keys:
        output = Right
    else:
        output = No
    return output


def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    while(True):
        
        if not paused:
            screen = screenshot(region=(0,40,2560,1120))
            last_time = time.time()
            screen = cv2.resize(screen, (450,260))
            
            keys = key_check()
            output = output_keys(keys)
            training_data.append([screen,output])

            last_time = time.time()
            cv2.imshow('window',cv2.resize(screen,(640,360)))
            if cv2.waitKey(25) & 0xFF == ord('y'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 100 == 0:
                print(len(training_data))
                
                if len(training_data) == 500:
                    np.save(file_name,training_data)
                    training_data = []
                    starting_value += .5
                    file_name = 'X:/project/training_data-{}.npy'.format(starting_value)

                    
        keys = key_check()


main(file_name, starting_value)
