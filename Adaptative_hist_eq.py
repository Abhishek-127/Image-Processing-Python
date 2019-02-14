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
import time

# Function to perform contrast enhancement using adaptive histogram equalization
#   Ref(s):
def histeqADAPT(im, radius=20):
    img = np.zeros(im.shape)
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            block = im[max(i-radius,0):min(i+radius,im.shape[0]), max(j-radius,0):min(j+radius,im.shape[1])]
            hst,bins = np.histogram(block.flatten(),256, (0,255))
            # Calculate the cumulative histogram
            cdf = hst.cumsum()        
            # Normalize the CDF
            cdf = 255 * cdf / cdf[-1]
            img[i][j] = cdf[im[i][j]]
        print(".")
    return img

# Function to read in an image in grayscale and
# store it in an np array.
def imread_gray(fname):
    img = PIL.Image.open(fname)
    return np.asarray(img)

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

    from scipy.misc import toimage
    toimage(img).save(fname)
    toimage(img).show()

    img_uint8 = np.uint8(img)
    imgSv = PIL.Image.fromarray(img_uint8,'L')
    imgSv.save(fname)


def main():
    start = time.time()
    img_name = sys.argv[1]
    img = imread_gray(img_name)
    plot_IMGhist(img)
    results = histeqADAPT(img)
    end = time.time()
    time_taken = (end-start)
    plot_IMGhist(results)
    fname = img_name + "RESTORED.jpg"

    imwrite_gray(fname, results)
    print(time_taken)

if __name__ == '__main__':
    main()