################################
## Travis Allen
## CS 6640 Project 3 Problem 3
################################

## import necessary libraries
import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
from skimage import morphology
import matplotlib.pyplot as plt
import time
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

## start timer
tic = time.perf_counter()

## read some images
img_1_0 = io.imread("images/img_1_0.png",as_gray=True)
img_1_1 = io.imread("images/img_1_1.png",as_gray=True)

## find their phase correlation
pc = phase_correlation(img_1_0,img_1_1)

## there is some numerical error so imaginary parts != 0. Extract just the real part
size = np.shape(pc)
pc_real = np.zeros((size[0],size[1]))
pc_real = pc.real

## threshold to see results a bit clearer
@jit(nopython=True)
def threshold(image):
    ## set parameter below which to send intensities to 0
    cutoff = 0.94

    ## determine the max intensity in the image
    max_intensity = np.amax(image)

    ## normalize the image
    thresholded_image = image/max_intensity

    ## Loop through each pixel and adjust intensity depending on how close it is to max intensity
    size = np.shape(image)
    for i in range(size[0]):
        for j in range(size[1]):
            if (image[i,j] <= cutoff*max_intensity):
                thresholded_image[i,j] = 0
            else:
                thresholded_image[i,j] = 1

    return thresholded_image

t_image = threshold(pc_real)

## dilate to make even clearer
t_image = morphology.dilation(t_image)

## print the maximum element of pc_real
# print(np.amax(pc_real))

## find the indices of the maximal element of the phase correlation, call them lambda_y and lambda_x
index = np.unravel_index(np.argmax(pc_real, axis=None), pc_real.shape)
lambda_y = index[0]
lambda_x = index[1]

## now compute the phase correlations of each region
m = size[0]
n = size[1]
ml = m-lambda_y
nl = n-lambda_x
max_correlation = np.zeros((4,1))

## region 0 ------------------------------------------------------------------------
region0_image1 = np.zeros((lambda_y,lambda_x))
region0_image2 = np.zeros((lambda_y,lambda_x))

## populate according to convention described in report
region0_image1[0:-1,0:-1] = img_1_0[ml:-1,nl:-1]
region0_image2[0:-1,0:-1] = img_1_1[0:lambda_y-1,0:lambda_x-1]

## compute phase correlation
region_0_phase_correlation = phase_correlation(region0_image1,region0_image2)

## find the maximal element of the phase correlation and store it
max_correlation[0] = np.amax(region_0_phase_correlation.real)

## region 1 ------------------------------------------------------------------------
region1_image1 = np.zeros((lambda_y,nl))
region1_image2 = np.zeros((lambda_y,nl))

## populate according to convention described in report
region1_image1[0:-1,0:-1] = img_1_0[ml:-1,0:nl-1]
region1_image2[0:-1,0:-1] = img_1_1[0:lambda_y-1,lambda_x:-1]

## compute phase correlation
region_1_phase_correlation = phase_correlation(region1_image1,region1_image2)

## find the maximal element of the phase correlation and store it
max_correlation[1] = np.amax(region_1_phase_correlation.real)

## region 2 ------------------------------------------------------------------------
region2_image1 = np.zeros((ml,lambda_x))
region2_image2 = np.zeros((ml,lambda_x))

## populate according to convention described in report
region2_image1[0:-1,0:-1] = img_1_0[0:ml-1,nl:-1]
region2_image2[0:-1,0:-1] = img_1_1[lambda_y:-1,0:lambda_x-1]

## compute phase correlation
region_2_phase_correlation = phase_correlation(region2_image1,region2_image2)

## find the maximal element of the phase correlation and store it
max_correlation[2] = np.amax(region_2_phase_correlation.real)

## region 3 ------------------------------------------------------------------------
region3_image1 = np.zeros((ml,nl))
region3_image2 = np.zeros((ml,nl))

## populate according to convention described in report
region3_image1[0:-1,0:-1] = img_1_0[0:ml-1,0:nl-1]
region3_image2[0:-1,0:-1] = img_1_1[lambda_y:-1,lambda_x:-1]

## compute phase correlation
region_3_phase_correlation = phase_correlation(region3_image1,region3_image2)

## find the maximal element of the phase correlation and store it
max_correlation[3] = np.amax(region_3_phase_correlation.real)

## now find the index of the maximum correlation. This will correspond to the region.
max_idx = np.unravel_index(np.argmax(max_correlation, axis=None), max_correlation.shape)

# end timer
toc = time.perf_counter()
print("execution time: ",toc-tic)

## now create the canvas depending on the index. 
if (max_idx[0] == 0):
    ## create canvas
    canvas = np.ones((2*m-lambda_y,2*n-lambda_x))*127

    ## now plot the images in the right places
    canvas[0:m,0:n] = img_1_0
    canvas[ml:ml+m,nl:nl+n] = img_1_1
    
    plt.figure(figsize=(9,9.5))
    plt.imshow(canvas,cmap='gray')
    plt.axis('off')
    plt.title("Two-image Mosaic")
    plt.tight_layout()
    plt.savefig("p3_output/img_1_mosaic_lp.png")
    plt.show()

    plt.figure(figsize=(12,6.5))
    plt.subplot(1,2,1)
    plt.imshow(img_1_0,cmap='gray')
    plt.axis('off')
    plt.title("Image 1")

    plt.subplot(1,2,2)
    plt.imshow(img_1_1,cmap='gray')
    plt.axis('off')
    plt.title("Image 2")

    plt.tight_layout()
    # plt.savefig("p3_output/img_0_pieces.png")
    plt.show()

    print("Region 0")

elif (max_idx[0] == 1):
    ## create canvas
    canvas = np.ones((2*m-lambda_y,n+lambda_x))*127

    ## now plot the images in the right places
    canvas[ml:-1,0:n] = img_1_0
    canvas[0:m,lambda_x:-1] = img_1_1
    
    plt.figure(figsize=(9,9.5))
    plt.imshow(canvas,cmap='gray')
    plt.axis('off')
    plt.title("Two-image Mosaic")
    plt.tight_layout()
    plt.savefig("p3_output/img_1_mosaic_lp.png")
    plt.show()

    plt.figure(figsize=(12,6.5))
    plt.subplot(1,2,1)
    plt.imshow(img_1_0,cmap='gray')
    plt.axis('off')
    plt.title("Image 1")

    plt.subplot(1,2,2)
    plt.imshow(img_1_1,cmap='gray')
    plt.axis('off')
    plt.title("Image 2")

    plt.tight_layout()
    # plt.savefig("p3_output/img_0_pieces.png")
    plt.show()

    print("Region 1")

elif (max_idx[0] == 2):
    ## create canvas
    canvas = np.ones((m+lambda_y,2*n-lambda_x))*127

    ## now plot the images in the right places
    canvas[lambda_y-1:-1,0:n] = img_1_0
    canvas[0:m,nl-1:-1] = img_1_1
    plt.figure(figsize=(9,9.5))
    plt.imshow(canvas,cmap='gray')
    plt.axis('off')
    plt.title("Two-image Mosaic")
    plt.tight_layout()
    plt.savefig("p3_output/img_1_mosaic_lp.png")
    plt.show()

    plt.figure(figsize=(12,6.5))
    plt.subplot(1,2,1)
    plt.imshow(img_1_0,cmap='gray')
    plt.axis('off')
    plt.title("Image 1")

    plt.subplot(1,2,2)
    plt.imshow(img_1_1,cmap='gray')
    plt.axis('off')
    plt.title("Image 2")

    plt.tight_layout()
    # plt.savefig("p3_output/img_0_pieces.png")
    plt.show()

    print("Region 2")

elif (max_idx[0] == 3):
    ## create canvas
    canvas = np.ones((m+lambda_y,n+lambda_x))*127
 
    ## now plot the images in the right places
    canvas[lambda_y-1:-1,lambda_x-1:-1] = img_1_0
    canvas[0:m,0:n] = img_1_1
    
    plt.figure(figsize=(9,9.5))
    plt.imshow(canvas,cmap='gray')
    plt.axis('off')
    plt.title("Two-image Mosaic")
    plt.tight_layout()
    plt.savefig("p3_output/img_1_mosaic_lp.png")
    plt.show()

    plt.figure(figsize=(12,6.5))
    plt.subplot(1,2,1)
    plt.imshow(img_1_0,cmap='gray')
    plt.axis('off')
    plt.title("Image 1")

    plt.subplot(1,2,2)
    plt.imshow(img_1_1,cmap='gray')
    plt.axis('off')
    plt.title("Image 2")

    plt.tight_layout()
    # plt.savefig("p3_output/img_0_pieces.png")
    plt.show()

    print("Region 3")


