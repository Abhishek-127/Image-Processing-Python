# @author: Abhishek Jhoree
# @email: ajhoree@uoguelph.ca

# first algorithm
# Implementation of the Adaptative histogram equalization

import numpy as np
from scipy.constants.constants import pi
from numpy.ma.core import exp
import math
import scipy.ndimage as nd
import pylab
import PIL
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
import sys
from PIL import  Image
from scipy.misc import toimage
import scipy.misc

def SSIM(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(np.float)
    img_mat_2=img_mat_2.astype(np.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=np.average(ssim_map)

    return index

def imread_gray(fname):
    img = PIL.Image.open(fname)
    return np.asarray(img)

# Function to perform contrast enhancement using image histogram equalization
def histeq(im,nbr_bins=256):
    img = np.zeros(im.shape,dtype=np.int16)
    
    # Calculate the histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,(0,255),density=False)
    
    # Calculate the probability density function of the sub-histogram
    histpdf = imhist / np.float32(im.size)
    # Derive the cumulative histogram
    cdf = histpdf.cumsum()
    
    # Histogram equalization for each of the sub-histograms
    fL = 255 * cdf
    fL = fL.astype('uint8')
    print(fL)
    
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            img[i][j] = fL[im[i][j]]

    xr = np.arange(0,256,1)
    pylab.plot(xr,fL)
    pylab.show()

    return img, imhist

# Function to display the histogram of an image 
def plot_IMGhist(img,nbr_bins=256):    
    # the histogram of the data
    plt.hist(img.flatten(),nbr_bins,(0,nbr_bins-1))
    print("Flatten")
    print(img.flatten())
    print(len(img.flatten()) )
    plt.xlabel('Graylevels')
    plt.ylabel('No. Pixels')
    plt.title('Intensity Histogram')
    plt.grid(True)

    plt.show()

# Function to display an image histogram
def plot_hist(hst,nbr_bins=256):    

    xr = np.arange(0,nbr_bins,1)
    pylab.plot(xr,hst)
    
    pylab.show()

def imwrite_gray(fname,img):
    # w, h = 512, 512
    # img = np.zeros((h, w, 3), dtype=np.uint8)
    # img[256, 256] = [255, 0, 0]
    # img = Image.fromarray(img, 'RGB')
    # img.show()
    from scipy.misc import toimage
    toimage(img).save("arch.png")
    toimage(img).show()

    img_uint8 = np.uint8(img)
    imgSv = PIL.Image.fromarray(img_uint8,'L')
    imgSv.save(fname)


def main():
    img_name = sys.argv[1]
    img = imread_gray(img_name)
    plot_IMGhist(img)
    results = histeq(img)

    plot_IMGhist(results[0])

    # toimage(results[1]).show()
    fname = "Abhishek.jpg"
    # array_resized_image = scipy.misc.imresize(results[0], (255, 255), interp='nearest', mode=None)
    print("Befoore ")
    imwrite_gray(fname, results[0]) # uncomment here
    index = SSIM(img, results[0])
    print(index)
    

if __name__ == '__main__':
    main()
