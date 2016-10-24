import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt

def linear_brightness():
    '''
    Reads the images and computes brightness by linear regression
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

    #pdb.set_trace()

    def img_crop(c_img):
        '''
        Crops image around the center
        Args: cv2 image object
        Return: cropped cv2 image object
        '''
        rows, columns, channel = c_img.shape
        nrows = rows/2
        ncolumns = columns/2
        return(c_img[(nrows - 500):(nrows + 500), (ncolumns - 500):(ncolumns + 500)])

    def img_avg(a_img):
        '''
        Averages the brightness of the image
        Args: cv2 image object
        Return: averaged float value
        '''
        rows, columns, channel = a_img.shape
        total_sum = 0.0
        # 0 - blue, 1 - green, 2 - red
        for i in range(rows):
            for j in range(columns):
                total_sum = total_sum + a_img[i,j,1]
        return(total_sum/(a_img.size))

    # image 1
    nimg1 = img_crop(img1)
    aimg1 = img_avg(nimg1)

    # image 2
    nimg2 = img_crop(img2)
    aimg2 = img_avg(nimg2)

    # image 3
    nimg3 = img_crop(img3)
    aimg3 = img_avg(nimg3)

    # image 4
    nimg4 = img_crop(img4)
    aimg4 = img_avg(nimg4)

    # image 5
    nimg5 = img_crop(img5)
    aimg5 = img_avg(nimg5)

    # image 6
    nimg6 = img_crop(img6)
    aimg6 = img_avg(nimg6)

    # image 7
    nimg7 = img_crop(img7)
    aimg7 = img_avg(nimg7)

    # image 8
    nimg8 = img_crop(img8)
    aimg8 = img_avg(nimg8)

    # x and y axis
    B = [aimg1, aimg2, aimg3, aimg4, aimg5, aimg6, aimg7, aimg8]
    T = [1.0/350.0, 1.0/250.0, 1.0/180.0, 1.0/125.0, 1.0/90.0, 1.0/60.0, 1.0/45.0, 1.0/30.0]

    # logs
    log_B = np.log10(B)
    log_T = np.log10(T)

    # mean
    mean_log_B = (np.sum(log_B))/float(len(log_B))
    mean_log_T = (np.sum(log_T))/float(len(log_T))

    def linear_regression(x, y, x_bar, y_bar):
        '''
        Returns the linearized Y axis values
        Args: All logarithmic values of B, T and their means
        Return: Least squares estimate of B
        '''
        num_r = 0
        den1_r = 0
        den2_r = 0
        for i in range(8):
            num_r = num_r + ((x[i] - x_bar)*(y[i] - y_bar))
            den1_r = den1_r + ((x[i] - x_bar) ** 2)
            den2_r = den2_r + ((y[i] - y_bar) ** 2)

        den3_r = (den1_r * den2_r) ** (0.5)

        # pearson's coefficient
        r = num_r/den3_r

        # standard deviation
        sx = (den1_r/ float(len(x) - 1)) ** (0.5)
        sy = (den2_r/ float(len(y) - 1)) ** (0.5)

        # slope
        b1 = r*(sy/sx)

        # intercept
        log_k = y_bar - b1*x_bar

        return(log_k + (b1)*log_T)

    # linear regression on log_B
    linearized_log_B = linear_regression(log_T, log_B, mean_log_T, mean_log_B)

    # plotting
    plt.plot(log_T, linearized_log_B)
    plt.plot(log_T, log_B)
    plt.scatter(mean_log_T, mean_log_B)
    plt.xlabel('Logarithm of Exposure Time')
    plt.ylabel('Logarithm of Brightness')
    plt.show()

def main():
    linear_brightness()

if __name__ == "__main__":
    main()
