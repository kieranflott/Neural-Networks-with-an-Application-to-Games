import numpy as np
from PIL import ImageGrab
import cv2
import time


def Edges(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[2],coords[3]), (coords[4],coords[5]), [255,255,255], 3)
    except:
        pass


def Driving_Area(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(processed, (3,3), 0 )
    processed = cv2.Canny(processed, threshold1=400, threshold2=500)
    vertices = np.array([[15,500],[15,300], [350,200], [550,225], [800,300], [800,500]], np.int32)
    processed = Driving_Area(processed, [vertices])

    # detects the edges
    lines = cv2.HoughLinesP(processed, 1, np.pi/180, 180,np.array([]), 200, 25)
    Edges(processed,lines)
    return processed


def main():
    last_time = time.time()
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(0,40, 850, 680)))
        new_screen = process_img(screen)
        print('Screen shot Took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('y'):
            cv2.destroyAllWindows()
            break

main()
