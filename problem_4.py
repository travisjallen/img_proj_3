################################
## Travis Allen
## CS 6640 Project 3 Problem 4
################################

from matplotlib import cm
import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
from skimage import morphology
import matplotlib.pyplot as plt
from numba import jit

#********************************************************#
#********************************************************#
#********************************************************#
#********************************************************#
## Define the max dimensions of any image we will use
max_rows = 510
max_cols = 510

#********************************************************#
#********************************************************#
#********************************************************#
#********************************************************#


## define the functions we will use

## define dataloader
def dataloader(folder_name,file_name):
    """Reads file file_name which contains names of images in folder_name in a new-line separated text file. Stores the images with those names in a 3D array"""
    sep = '/'
    path = sep.join([folder_name,file_name])
    ## read the file names
    with open(path) as f:
        lines = f.read().split('\n')

    ## figure out how many there are
    num_images = int(len(lines))

    ## create an array to store them
    img_array = np.zeros((max_rows,max_cols,num_images))
    
    ## fill it with the images
    for i in range(num_images):
        field = sep.join(['cell_images',lines[i]])
        img_array[:,:,i] = io.imread(field,as_gray=True)
    return img_array

## define function to perform phase correlation
def phase_correlation(F,G):
    """Returns real part of phase correlation of two images"""
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

    ## we only want the real part of the phase correlation. return it
    size = np.shape(correlation)
    correlation_real = np.zeros((size[0],size[1]))
    correlation_real = correlation.real

    return correlation_real

## define function to maximum of phase correlation
def max_phase_correlation(F,G):
    """Computes the maximal element in the phase correlation of two images"""
    ## compute phase correlation of the two images
    correlation = phase_correlation(F,G)

    ## find maxiumum phase correlation, return it
    maximum_phase_correlation = np.amax(correlation)
    return maximum_phase_correlation

def overlap(pc_real,F,G):
    ## find the indices of the maximal element of the phase correlation, call them lambda_y and lambda_x
    index = np.unravel_index(np.argmax(pc_real, axis=None), pc_real.shape)
    lambda_y = index[0]
    lambda_x = index[1]

    ## now compute the phase correlations of each region
    size = np.shape(pc_real)
    m = size[0]
    n = size[1]
    ml = m-lambda_y
    nl = n-lambda_x
    max_correlation = np.zeros((4,1))

    ## region 0 ------------------------------------------------------------------------
    region0_image1 = np.zeros((lambda_y,lambda_x))
    region0_image2 = np.zeros((lambda_y,lambda_x))
    if (lambda_x > 0 and lambda_y > 0):
        ## populate according to convention described in report
        region0_image1[0:-1,0:-1] = G[ml:-1,nl:-1]
        region0_image2[0:-1,0:-1] = F[0:lambda_y-1,0:lambda_x-1]

        ## compute phase correlation if region is larger than 0
        
        region_0_phase_correlation = phase_correlation(region0_image1,region0_image2)

        ## find the maximal element of the phase correlation and store it
        max_correlation[0] = np.amax(region_0_phase_correlation.real)
    else:
        max_correlation[0] = 0

    ## region 1 ------------------------------------------------------------------------
    region1_image1 = np.zeros((lambda_y,nl))
    region1_image2 = np.zeros((lambda_y,nl))
    
    if (lambda_x > 0 and lambda_y > 0):
        ## populate according to convention described in report
        region1_image1[0:-1,0:-1] = G[ml:-1,0:nl-1]
        region1_image2[0:-1,0:-1] = F[0:lambda_y-1,lambda_x:-1]

        ## compute phase correlation
        region_1_phase_correlation = phase_correlation(region1_image1,region1_image2)

        ## find the maximal element of the phase correlation and store it
        max_correlation[1] = np.amax(region_1_phase_correlation.real)

    else:
        max_correlation[1] = 0

    ## region 2 ------------------------------------------------------------------------
    region2_image1 = np.zeros((ml,lambda_x))
    region2_image2 = np.zeros((ml,lambda_x))

    if (lambda_x > 0 and lambda_y > 0):
        ## populate according to convention described in report
        region2_image1[0:-1,0:-1] = G[0:ml-1,nl:-1]
        region2_image2[0:-1,0:-1] = F[lambda_y:-1,0:lambda_x-1]

        ## compute phase correlation
        region_2_phase_correlation = phase_correlation(region2_image1,region2_image2)

        ## find the maximal element of the phase correlation and store it
        max_correlation[2] = np.amax(region_2_phase_correlation.real)
    else:
        max_correlation[2] = 0

    ## region 3 ------------------------------------------------------------------------
    region3_image1 = np.zeros((ml,nl))
    region3_image2 = np.zeros((ml,nl))

    ## populate according to convention described in report
    region3_image1[0:-1,0:-1] = G[0:ml-1,0:nl-1]
    region3_image2[0:-1,0:-1] = F[lambda_y:-1,lambda_x:-1]

    ## compute phase correlation
    region_3_phase_correlation = phase_correlation(region3_image1,region3_image2)

    ## find the maximal element of the phase correlation and store it
    max_correlation[3] = np.amax(region_3_phase_correlation.real)

    ## now find the index of the maximum correlation. This will correspond to the region.
    max_idx = np.unravel_index(np.argmax(max_correlation, axis=None), max_correlation.shape)

    ## now create the canvas depending on the index. 
    if (max_idx[0] == 0):
        ## create canvas
        # canvas = np.ones((2*m-lambda_y,2*n-lambda_x))*127

        ## now plot the images in the right places
        # canvas[0:m,0:n] = G
        # G_index = [m,n]
        
        ## how far away from the origin of image G do we place image F
        F_index = [ml,nl]

    elif (max_idx[0] == 1):
        ## create canvas
        # canvas = np.ones((2*m-lambda_y,n+lambda_x))*127

        ## now plot the images in the right places
        # canvas[ml:-1,0:n] = G
        # G_index = [ml,0]
        # canvas[0:m,lambda_x:-1] = F
        # F_index = [0,lambda_x]

        ## how far away from the origin of image G do we place image F
        F_index = [-ml,lambda_x]
        
    elif (max_idx[0] == 2):
        ## create canvas
        # canvas = np.ones((m+lambda_y,2*n-lambda_x))*127

        ## now plot the images in the right places
        # canvas[lambda_y-1:-1,0:n] = G
        # G_index = [lambda_y-1,0]
        # canvas[0:m,nl-1:-1] = F
        # F_index = [0,nl]
        
        ## how far away from the origin of image G do we place image F
        F_index = [lambda_y,-nl]
        
        
    elif (max_idx[0] == 3):
        ## create canvas
        # canvas = np.ones((m+lambda_y,n+lambda_x))*127
    
        ## now plot the images in the right places
        # canvas[lambda_y-1:-1,lambda_x-1:-1] = G
        # G_index = [lambda_y-1,lambda_x-1]
        # canvas[0:m,0:n] = F
        # F_index = [0,0]

        ## how far away from the origin of image G do we place image F
        F_index = [-lambda_y,-lambda_x]

    # return lambdas
    return F_index
        
## read the data in cell_images/read_cells.txt
img_array = dataloader("cell_images","read_cells.txt")

## plot the images
plt.subplot(2,3,1)
plt.imshow(img_array[:,:,0],cmap='gray')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(img_array[:,:,1],cmap='gray')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(img_array[:,:,2],cmap='gray')
plt.axis('off')
plt.subplot(2,3,4)
plt.imshow(img_array[:,:,3],cmap='gray')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(img_array[:,:,4],cmap='gray')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(img_array[:,:,5],cmap='gray')
plt.axis('off')
plt.suptitle('Cell Images')
plt.tight_layout()
plt.savefig('images/cell_images_subplot.png')
plt.show()


# ## determine if they are the same size
# if ((np.amax(sizes[:,0]) != np.amin(sizes[:,0])) or (np.amax(sizes[:,1]) != np.amin(sizes[:,1]))):
#     ## if this is the case, then at least one image is not as big as the biggest image
#     print("Error: images not of same dimension")
# else:
#     ## execute as planned
#     ## create a 3D array to store the images
#     img_array = np.zeros((int(sizes[0,0]),int(sizes[0,1]),6))
    
#     ## fill it with the images
#     img_array[:,:,0] = img_0
#     img_array[:,:,1] = img_1
#     img_array[:,:,2] = img_2
#     img_array[:,:,3] = img_3
#     img_array[:,:,4] = img_4
#     img_array[:,:,5] = img_5

    