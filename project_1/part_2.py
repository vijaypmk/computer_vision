import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
import pdb

class project:

    def img_crop(self, c_img):
        '''
        Crops image around the center
        Args: cv2 image object
        Return: cropped cv2 image object
        '''
        rows, columns, channel = c_img.shape
        nrows = rows/2
        ncolumns = columns/2
        return(c_img[(nrows - 500):(nrows + 500), (ncolumns - 500):(ncolumns + 500)])

    def img_avg(self, a_img, color_channel):
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
                total_sum = total_sum + a_img[i, j, color_channel]

        return(total_sum/(rows*columns))

    def part_1_init(self, color_channel):
        '''
        Initializes images and computes average brightness
        Args: None
        Return: Brightness (B), Exposure time (T), log B, logT, mean B, mean T
        '''
        img1 = cv2.imread('WP_350.JPG')
        img2 = cv2.imread('WP_250.JPG')
        img3 = cv2.imread('WP_180.JPG')
        img4 = cv2.imread('WP_125.JPG')
        img5 = cv2.imread('WP_90.JPG')
        img6 = cv2.imread('WP_60.JPG')
        img7 = cv2.imread('WP_45.JPG')
        img8 = cv2.imread('WP_30.JPG')

        # image 1
        nimg1 = self.img_crop(img1)
        aimg1 = self.img_avg(nimg1, color_channel)

        # image 2
        nimg2 = self.img_crop(img2)
        aimg2 = self.img_avg(nimg2, color_channel)

        # image 3
        nimg3 = self.img_crop(img3)
        aimg3 = self.img_avg(nimg3, color_channel)

        # image 4
        nimg4 = self.img_crop(img4)
        aimg4 = self.img_avg(nimg4, color_channel)

        # image 5
        nimg5 = self.img_crop(img5)
        aimg5 = self.img_avg(nimg5, color_channel)

        # image 6
        nimg6 = self.img_crop(img6)
        aimg6 = self.img_avg(nimg6, color_channel)

        # image 7
        nimg7 = self.img_crop(img7)
        aimg7 = self.img_avg(nimg7, color_channel)

        # image 8
        nimg8 = self.img_crop(img8)
        aimg8 = self.img_avg(nimg8, color_channel)

        # x and y axis
        B = [aimg1, aimg2, aimg3, aimg4, aimg5, aimg6, aimg7, aimg8]
        T = [1.0/350.0, 1.0/250.0, 1.0/180.0, 1.0/125.0, 1.0/90.0, 1.0/60.0, 1.0/45.0, 1.0/30.0]

        # logs
        log_B = np.log10(B)
        log_T = np.log10(T)

        # mean
        mean_log_B = (np.sum(log_B))/float(len(log_B))
        mean_log_T = (np.sum(log_T))/float(len(log_T))

        return(B, T, log_B, log_T, mean_log_B, mean_log_T)

    def linearize_image(image, color_channel)
        '''
        Stores the computed value in the image
        Arguements: Image
        Return: Image
        '''
        B = self.part_2.init()
        avg = [self.img_avg(B[0], color_channel), self.img_avg(B[1], color_channel), self.img_avg(B[2], color_channel)]
        time = [1/1000, 1/500, 1/250]
        log_avg = np.log(avg)
        log_time = np.log(time)
        mean_log_avg = (np.sum(log_avg))/float(len(log_avg))
        mean_log_time = (np.sum(log_time))/float(len(log_time))


    def part_2_init(self, color_channel):
        '''
        Reads the different images
        Arguements: None
        Return: Image array
        '''
        #image 1/1000s
        img_1 = cv2.imread('1000s.jpg')

        #image 1/500s
        img_2 = cv2.imread('500s.jpg')

        #image 1/250s
        img_3 = cv2.imread('250s.jpg')

        B = [img_1, img_2, img_3]

        return B

    def part_2(self):
        '''
        Performslinearization
        Arguements: None
        Return: Image Array
        '''
        B = self.part_2_init()

        print(B[1][10,10,1])

        return(B[1][10,10,1])

    def linear_regression(self, x, y, x_bar, y_bar):
        '''
        Returns the linearized Y axis values
        Args: All logarithmic values of B, T and their means
        Return: Least squares estimate of log B, 1/g
        '''
        num_r = 0
        den1_r = 0
        den2_r = 0
        for i in range(len(x)):
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

        return(log_k + (b1)*x, b1)

    def part_1(self, color_channel):
        B, T, log_B, log_T, mean_log_B, mean_log_T = self.part_1_init(color_channel)

    # linear regression on log_B
        linearized_log_B, g_inv = self.linear_regression(log_T, log_B, mean_log_T, mean_log_B)

    # linearized B
        linearized_B = 10**(linearized_log_B)

    # plotting
        #plt.plot(log_T, linearized_log_B, label ='linearized estimate')
        #plt.plot(log_T, log_B, label = 'observed brightness')
        plt.plot(T, linearized_B**(1/g_inv))
        #plt.scatter(mean_log_T, mean_log_B))
        #plt.plot(T, B)
        plt.xlabel('Logarithm of Exposure Time')
        plt.ylabel('Linearized value of logarithm of Brightness')
        plt.savefig('log_T_vs_linearized_log_B_for_blue_channel')
        plt.show()

        return

def main():
    p = project()
    p.part_1(1)

if __name__ == "__main__":
    main()
