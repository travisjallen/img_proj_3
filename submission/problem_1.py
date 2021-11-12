#############################
## Travis Allen
## CS 6640 Project 3
#############################

## import necessary libraries
import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
import matplotlib.pyplot as plt


## read some images
img_3 = io.imread("images/img_3.png",as_gray=True)

## normalize the images
norm_img_3 = img_3/np.amax(img_3)

## compute the FDFT of the normalized images
img_3_f = np.fft.fft2(norm_img_3)

## "Retile" the FDFT images
img_3_f_retile = np.fft.fftshift(img_3_f)
# np.log(np.absolute(img_3_f_retile)**2)

## show results
# plt.figure(figsize=(12,9))
# plt.subplot(1,2,1)
# plt.imshow(img_3,cmap='gray')
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1,2,2)
# plt.imshow(np.log(np.absolute(img_3_f_retile)**2),cmap='gray')
# plt.title("Retiled Fourier Transform of Image")
# plt.axis('off')

# plt.tight_layout()
# plt.savefig("p1_output/img_3_ft_compare.png")
# plt.show()

## build fourier domain rect(u,v)
def rect_lowpass(image,scale,show_filter):
    ## image: Image array to be filtered
    ## scale: percent of each image dimension that will have a value of 1
    ## show_filter: boolean to decide whether or not to show the graph of the filter
    
    ## start with array of ones
    size = np.shape(image)
    rows = size[0]
    cols = size[1]
    filt = np.zeros((rows,cols))
    
    ## find coordinates of box corners 
    uly = int((rows/2)-(int((scale*rows)/2)))
    ulx = int((cols/2)-(int((scale*cols)/2)))
    lry = int((rows/2)+(int((scale*rows)/2)))
    lrx = int((cols/2)+(int((scale*cols)/2)))

    ## place box of ones in the right location
    filt[uly:lry,ulx:lrx] = 1

    if (show_filter == 1):
        plt.imshow(filt,cmap='gray')
        plt.axis('off')
        plt.title("rect(u,v) Filter")
        plt.tight_layout()
        plt.savefig("p1_output/rect_filter.png")
        plt.show()

    ## save the filter
    # io.imsave("images/gaussian_filter.png",filt_x*filt_y)

    ## multiply the filter with the fourier transform of the original image, return it
    return filt*image


## build fourier domain gaussian low pass filter #####################
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

    if (show_filter == 1):
        plt.figure(figsize=(12,6))
        plt.subplot(1,3,1)
        plt.imshow(filt_x,cmap='gray')
        plt.axis('off')
        plt.title("Gaussian in x")

        plt.subplot(1,3,2)
        plt.imshow(filt_y,cmap='gray')
        plt.axis('off')
        plt.title("Gaussian in y")

        plt.subplot(1,3,3)
        plt.imshow(filt_y*filt_x,cmap='gray')
        plt.axis('off')
        plt.title("Final Gaussian Filter")
        plt.tight_layout()
        plt.savefig("p1_output/gaussian_filter.png")
        plt.show()

    ## save the filter
    # io.imsave("images/gaussian_filter.png",filt_x*filt_y)

    ## multiply the filter with the fourier transform of the original image, return it
    return (filt_x*filt_y)*image

## do some filtering in the fourier domain
img_3_filter_f = rect_lowpass(img_3_f_retile,0.1,0)
# img_3_filter_f = gaussian_lowpass(img_3_f_retile,20,0)

# ## show the new power spectrum
# plt.imshow(np.log(np.absolute(img_3_filter_f)**2),cmap='gray')
# plt.title("Filtered power spectrum of Image")
# plt.axis('off')
# plt.show()

## compute inverse FFT
img_3_filter = np.fft.ifft2(img_3_filter_f)

## plot result
plt.figure(figsize=(12,9))
plt.subplot(1,2,1)
plt.imshow(norm_img_3,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(np.absolute(img_3_filter),cmap='gray')
plt.title('Fourier Low Pass Filtered Image, rect')
plt.axis('off')

plt.tight_layout()
plt.savefig("p1_output/img_3_f_rect.png")
plt.show()