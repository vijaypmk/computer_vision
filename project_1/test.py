import numpy as np
import cv2
import sys

img1 = cv2.imread('4184s.jpg')
img2 = cv2.imread('3106s.jpg')
img3 = cv2.imread('2000.jpg')
img5 = cv2.imread('1000s.jpg')
img4 = cv2.imread('1500s.jpg')
img6 = cv2.imread('750s.jpg')
img7 = cv2.imread('500s.jpg')
img8 = cv2.imread('350s.jpg')
img9 = cv2.imread('250s.jpg')
img10 = cv2.imread('125s.jpg')

B = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]

def count(image, channel):
    '''
    Counts the number of saturated pixels in the given image
    Arguements: Image
    Return: Count
    '''
    count=0
    rows, columns, channels = image.shape
    for i in range(rows):
        for j in range(columns):
            if(image[i,j, channel] == 255):
                count = count+1

    return count

def choose(B, channel, color):
    '''
    Calls the count function for each of the images
    Arguements: Array of images, channel
    Returns:    None
    '''
    for i in range(len(B)):
        count_blue = count(B[i], channel)
        print(count_blue, color)


    return

def main():
    choose(B, 0, "blue")
    choose(B, 1, "green")
    choose(B, 2, "red")

if __name__ == "__main__":
    main()
