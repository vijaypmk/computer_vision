import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt

def linear_brightness():
    '''
    Reads the images and computes brightness
    Args: None
    Return: None
    '''
    img1 = cv2.imread('IMG_1_350.JPG')
    img2 = cv2.imread('IMG_2_250.JPG')
    img3 = cv2.imread('IMG_3_180.JPG')
    img4 = cv2.imread('IMG_4_125.JPG')
    img5 = cv2.imread('IMG_5_90.JPG')
    img6 = cv2.imread('IMG_6_60.JPG')
    img7 = cv2.imread('IMG_7_45.JPG')
    img8 = cv2.imread('IMG_8_30.JPG')

    pdb.set_trace()

def main():
    linear_brightness()

if __name__ == "__main__":
    main()
