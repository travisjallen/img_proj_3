################################
## Travis Allen
## CS 6640 Project 3 Problem 4
################################

import numpy as np
import skimage as sk
from skimage import io
from skimage import filters
from skimage import morphology
import matplotlib.pyplot as plt

from numba import jit

## define the functions we will use
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

    ## populate according to convention described in report
    region0_image1[0:-1,0:-1] = G[ml:-1,nl:-1]
    region0_image2[0:-1,0:-1] = F[0:lambda_y-1,0:lambda_x-1]

    ## compute phase correlation
    region_0_phase_correlation = phase_correlation(region0_image1,region0_image2)

    ## find the maximal element of the phase correlation and store it
    max_correlation[0] = np.amax(region_0_phase_correlation.real)

    ## region 1 ------------------------------------------------------------------------
    region1_image1 = np.zeros((lambda_y,nl))
    region1_image2 = np.zeros((lambda_y,nl))

    ## populate according to convention described in report
    region1_image1[0:-1,0:-1] = G[ml:-1,0:nl-1]
    region1_image2[0:-1,0:-1] = F[0:lambda_y-1,lambda_x:-1]

    ## compute phase correlation
    region_1_phase_correlation = phase_correlation(region1_image1,region1_image2)

    ## find the maximal element of the phase correlation and store it
    max_correlation[1] = np.amax(region_1_phase_correlation.real)

    ## region 2 ------------------------------------------------------------------------
    region2_image1 = np.zeros((ml,lambda_x))
    region2_image2 = np.zeros((ml,lambda_x))

    ## populate according to convention described in report
    region2_image1[0:-1,0:-1] = G[0:ml-1,nl:-1]
    region2_image2[0:-1,0:-1] = F[lambda_y:-1,0:lambda_x-1]

    ## compute phase correlation
    region_2_phase_correlation = phase_correlation(region2_image1,region2_image2)

    ## find the maximal element of the phase correlation and store it
    max_correlation[2] = np.amax(region_2_phase_correlation.real)

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
        G_index = [m,n]
        # canvas[ml:ml+m,nl:nl+n] = F
        F_index = [ml,nl]

    elif (max_idx[0] == 1):
        ## create canvas
        # canvas = np.ones((2*m-lambda_y,n+lambda_x))*127

        ## now plot the images in the right places
        # canvas[ml:-1,0:n] = G
        G_index = [ml,0]
        # canvas[0:m,lambda_x:-1] = F
        F_index = [0,lambda_x]
        
    elif (max_idx[0] == 2):
        ## create canvas
        # canvas = np.ones((m+lambda_y,2*n-lambda_x))*127

        ## now plot the images in the right places
        # canvas[lambda_y-1:-1,0:n] = G
        G_index = [lambda_y-1,0]
        # canvas[0:m,nl-1:-1] = F
        F_index = [0,nl]
        
    elif (max_idx[0] == 3):
        ## create canvas
        # canvas = np.ones((m+lambda_y,n+lambda_x))*127
    
        ## now plot the images in the right places
        # canvas[lambda_y-1:-1,lambda_x-1:-1] = G
        G_index = [lambda_y-1,lambda_x-1]
        # canvas[0:m,0:n] = F
        F_index = [0,0]

    lambdas = np.zeros((1,2))
    lambdas[0,0] = lambda_y
    lambdas[0,1] = lambda_x
    # return lambdas
    return G_index,F_index
        
## read the images. this is a cumbersome way but it works
img_0 = io.imread('cell_images/0001.000.png',as_gray=True)
img_1 = io.imread('cell_images/0001.001.png',as_gray=True)
img_2 = io.imread('cell_images/0001.002.png',as_gray=True)
img_3 = io.imread('cell_images/0001.004.png',as_gray=True)
img_4 = io.imread('cell_images/0001.005.png',as_gray=True)
img_5 = io.imread('cell_images/0001.006.png',as_gray=True)

num_images = 6

## find the shape of each image
sizes = np.zeros((6,2))
sizes[0,:] = np.shape(img_0)
sizes[1,:] = np.shape(img_1)
sizes[2,:] = np.shape(img_2)
sizes[3,:] = np.shape(img_3)
sizes[4,:] = np.shape(img_4)
sizes[5,:] = np.shape(img_5)

## determine if they are the same size
if ((np.amax(sizes[:,0]) != np.amin(sizes[:,0])) or (np.amax(sizes[:,1]) != np.amin(sizes[:,1]))):
    ## if this is the case, then at least one image is not as big as the biggest image
    print("Error: images not of same dimension")
else:
    ## execute as planned
    ## create a 3D array to store the images
    img_array = np.zeros((int(sizes[0,0]),int(sizes[0,1]),6))
    
    ## fill it with the images
    img_array[:,:,0] = img_0
    img_array[:,:,1] = img_1
    img_array[:,:,2] = img_2
    img_array[:,:,3] = img_3
    img_array[:,:,4] = img_4
    img_array[:,:,5] = img_5

    ## Create array to store pair information
    pairs = np.zeros((10,5))

    ## compute the maximum phase correlation of each image wrt image 0
    max_0 = np.zeros((num_images-1,1))
    for i in range(num_images-1):
        max_0[i] = max_phase_correlation(img_array[:,:,0],img_array[:,:,i+1])
    
    ## find the index of the maximum of these maxima, as well as the value
    index_pair_0 = np.unravel_index(np.argmax(max_0, axis=None), max_0.shape)
    supremum = max_0[index_pair_0]

    ## store the pair information
    pairs[0,0] = 0                  ## base image
    pairs[0,1] = index_pair_0[0]    ## paired image
    pairs[0,2] = supremum           ## phase correlation supremum

    ## check to see if there are any other pairs with 0
    delta = 0.5*supremum
    counts = 1

    ## loop through maximum phase correlations with image 0. Dont need to include image 0
    for i in range((num_images-1)):
        ## if the maximum phase correlation is in the allowed interval and is not a previously added pair, add it to pairs
        if (max_0[i] > (supremum-delta) and (i !=pairs[0,1])):
            ## the image is a pair with image 0
            pairs[counts,0] = 0
            pairs[counts,1] = i+1
            pairs[counts,2] = max_0[i]
            counts += 1
    
    ## now compute the phase correlations with respect to image 1. Don't need to include image 0 or image 1
    max_1 = np.zeros((num_images-2,1))
    for i in range(num_images-2):
        max_1[i] = max_phase_correlation(img_array[:,:,1],img_array[:,:,i+2])

    for i in range((num_images-2)):
        ## if the maximum phase correlation is in the allowed interval 
        if (max_1[i] > (supremum-delta)):
            ## the image is a pair with image 1
            pairs[counts,0] = 1
            pairs[counts,1] = i+2
            pairs[counts,2] = max_1[i]
            counts += 1
    
    ## now compute the phase correlations with respect to image 2. Don't need to include image 0, 1, or 2
    max_2 = np.zeros((num_images-3,1))
    for i in range(num_images-3):
        max_2[i] = max_phase_correlation(img_array[:,:,2],img_array[:,:,i+3])

    for i in range((num_images-3)):
        ## if the maximum phase correlation is in the allowed interval 
        if (max_2[i] > (supremum-delta)):
            ## the image is a pair with image 1
            pairs[counts,0] = 2
            pairs[counts,1] = i+3
            pairs[counts,2] = max_2[i]
            counts += 1
    
    ## now compute the phase correlations with respect to image 3. Don't need to include image 0, 1, 2, or 3
    max_3 = np.zeros((num_images-4,1))
    for i in range(num_images-4):
        max_3[i] = max_phase_correlation(img_array[:,:,3],img_array[:,:,i+4])

    for i in range((num_images-4)):
        ## if the maximum phase correlation is in the allowed interval 
        if (max_3[i] > (supremum-delta)):
            ## the image is a pair with image 1
            pairs[counts,0] = 3
            pairs[counts,1] = i+4
            pairs[counts,2] = max_3[i]
            counts += 1

    ## now compute the phase correlations with respect to image 4. Don't need to include image 0, 1, 2, 3, 4
    max_4 = np.zeros((num_images-5,1))
    for i in range(num_images-5):
        max_4[i] = max_phase_correlation(img_array[:,:,4],img_array[:,:,i+5])

    for i in range((num_images-5)):
        ## if the maximum phase correlation is in the allowed interval 
        if (max_4[i] > (supremum-delta)):
            ## the image is a pair with image 1
            pairs[counts,0] = 4
            pairs[counts,1] = i+5
            pairs[counts,2] = max_4[i]
            counts += 1
    
    ## now we have found all of the pairs. Find the number of nonzero rows in pairs
    pair_shape = np.shape(pairs)
    row_count = 1
    for i in range(pair_shape[1]):
        if (pairs[i,2] != 0):
            row_count +=1

    ## build new pairs array, fill it
    pairs_exact = np.zeros((row_count,5))
    pairs_exact[:,:] = pairs[0:row_count,:]
    # print(pairs_exact)
    
    ## define a canvas
    canvas_width = 10*np.amax(sizes[:,1])
    canvas_height = 10*np.amax(sizes[:,0])
    canvas = np.ones((canvas_height,canvas_width))

    ## now find the overlap of each pair
    for i in range(row_count):
        ## compute phase correlation of image in col 0 and col 1 at this row of pairs_exact
        pc = phase_correlation(img_array[:,:,int(pairs_exact[i,0])],img_array[:,:,int(pairs_exact[i,1])])

        ## Determine the overlap condition
        G_I,F_I = overlap(pc,img_array[:,:,int(pairs_exact[i,0])],img_array[:,:,int(pairs_exact[i,1])])
        