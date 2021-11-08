################################
## Travis Allen
## CS 6640 Project 3 Problem 2
################################

## import necessary libraries
import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
from skimage import morphology
import matplotlib.pyplot as plt

from numba import jit

## define function to perform phase correlation
def phase_correlation(F,G):
    ## compute fourier transform of image G
    G_fourier = np.fft.fft2(G)

    ## compute complex conjugate of fourier transform of image F 
    F_conj_fourier = np.fft.fft2(F).conj()

    ## compute F*G/(|F*G|)
    correlation_fourier = F_conj_fourier*G_fourier/(np.absolute(F_conj_fourier*G_fourier))

    ## low pass filter the result
    # correlation_fourier = gaussian_lowpass(correlation_fourier,20,0)

    ## compute inverse fourier transform to find phase correlation
    correlation = (np.fft.ifft2(correlation_fourier)) ## np.fft.fftshift
    return correlation

## define function to perform low pass filtering in fourier domain, to be used in phase correlation
@jit(nopython=True)
def gaussian_lowpass(image,cutoff,show_filter):
    ## image: Image array to be filtered
    ## cutoff: number of standard deviations to include. Typical value is 8
    ## show_filter: boolean to decide whether or not to show the graph of the filter
    
    ## start with array of ones
    size = np.shape(image)
    rows = size[0]
    cols = size[1]
    filt_x = np.ones((rows,cols))
    filt_y = np.ones((rows,cols))

    ## elementwise multiply each directional array of ones with gaussian
    sigma_x = cols/cutoff
    sigma_y = rows/cutoff

    ## y direction
    for i in range(rows):
        for j in range(cols):
            ## transform array index to spatial coordindate
            spatial_i = i - ((rows/2)-1)
            spatial_j = j - ((cols/2)-1)
            
            ## calculate appropriate value of gaussian, replace the appropriate element
            filt_y[i,j] = (1/(np.sqrt(2*np.pi)*sigma_y))*np.exp(-0.5*((spatial_i/sigma_y)**2))
            filt_x[i,j] = (1/(np.sqrt(2*np.pi)*sigma_x))*np.exp(-0.5*((spatial_j/sigma_x)**2))

    # if (show_filter == 1):
    #     plt.figure(figsize=(12,6))
    #     plt.subplot(1,3,1)
    #     plt.imshow(filt_x,cmap='gray')
    #     plt.axis('off')
    #     plt.title("Gaussian in x")

    #     plt.subplot(1,3,2)
    #     plt.imshow(filt_y,cmap='gray')
    #     plt.axis('off')
    #     plt.title("Gaussian in y")

    #     plt.subplot(1,3,3)
    #     plt.imshow(filt_y*filt_x,cmap='gray')
    #     plt.axis('off')
    #     plt.title("Final Gaussian Filter")
    #     plt.tight_layout()
    #     plt.savefig("p1_output/gaussian_filter.png")
    #     plt.show()

    ## save the filter
    # io.imsave("images/gaussian_filter.png",filt_x*filt_y)

    ## multiply the filter with the fourier transform of the original image, return it
    return (filt_x*filt_y)*image

## read some images
img_0_0 = io.imread("images/img_0_0.png",as_gray=True)
img_0_1 = io.imread("images/img_0_1.png",as_gray=True)

## find their phase correlation
pc = phase_correlation(img_0_0,img_0_1)

## there is some numerical error so imaginary parts != 0. Extract just the real part
size = np.shape(pc)
pc_real = np.zeros((size[0],size[1]))

pc_real = pc.real

print(np.amax(pc_real))
print(np.amin(pc_real))

## threshold to see results a bit clearer
@jit(nopython=True)
def threshold(image):
    ## set parameter below which to send intensities to 0
    cutoff = 0.8

    ## determine the max intensity in the image
    max_intensity = np.amax(image)

    ## normalize the image
    thresholded_image = image/max_intensity

    ## Loop through each pixel and adjust intensity depending on how close it is to max intensity
    size = np.shape(image)
    for i in range(size[0]):
        for j in range(size[1]):
            if (image[i,j] < cutoff*max_intensity):
                thresholded_image[i,j] = 0
            else:
                thresholded_image[i,j] = 1

    return thresholded_image

t_image = threshold(pc_real)

t_image = morphology.dilation(t_image)

plt.figure(figsize=(12,12.5))
plt.subplot(2,2,1)
plt.imshow(img_0_0,cmap='gray')
plt.axis('off')
plt.title("Image 0")

plt.subplot(2,2,2)
plt.imshow(img_0_1,cmap='gray')
plt.axis('off')
plt.title("Image 1")

plt.subplot(2,2,3)
plt.imshow(pc_real,cmap='gray')
plt.axis('off')
plt.title("Raw Phase Correlation")

plt.subplot(2,2,4)
plt.imshow(t_image,cmap='gray')
plt.axis('off')
plt.title("Thresholded Phase Correlation")

plt.tight_layout()
plt.savefig("p2_output/phase_correlation_raw.png")
plt.show()