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

# write the name of the txt file that contains the list of 
# images and the path from the current directory to the 
# folder with the images and list
list_file_name = "read_cells.txt"
image_folder_path = "cell_images"

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
        field = sep.join([folder_name,lines[i]])
        img_array[:,:,i] = io.imread(field,as_gray=True)
    return img_array,num_images

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

    ## multiply the filter with the fourier transform of the original image, return it
    return (filt_x*filt_y)*image

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
    # correlation_fourier = gaussian_lowpass(correlation_fourier,50,0)

    ## compute inverse fourier transform to find phase correlation
    correlation = (np.fft.ifft2(correlation_fourier)) ## np.fft.fftshift
    (rows,cols) = np.shape(correlation)
    correlation_real = np.zeros((rows,cols))
    correlation_real = correlation.real
    return correlation_real
        
## read the data in cell_images/read_cells.txt
img_array,num_images = dataloader(image_folder_path,list_file_name)

## make canvas
canvas = np.ones((5*max_rows,5*max_cols))
center_y = int(2.5*max_rows)
center_x = int(2.5*max_cols)

# ## compute phase correlation of image 0 and image 1
# pc = phase_correlation(img_array[:,:,1],img_array[:,:,0])

# # plt.imshow(pc,cmap='gray')
# # plt.show()

# ## find the index of the max
# index = np.unravel_index(np.argmax(pc, axis=None), pc.shape)

# ## compute new origin coords
# origin_y = center_y - max_rows + index[0]
# origin_x = center_x + index[1]
# terminus_y = origin_y + max_rows
# terminus_x = origin_x + max_cols

# ## make canvas
# canvas = np.ones((6*max_rows,6*max_cols))
# center_y = 3*max_rows
# center_x = 3*max_cols

# ## normalize the images
# for i in range(num_images):
#     img_array[:,:,i] = img_array[:,:,i]/np.amax(img_array[:,:,i])

# ## place image 0
# canvas[center_y:int(center_y + 510),center_x:int(center_x + 510)] = img_array[:,:,0]

# ## place image 1
# canvas[origin_y:terminus_y,origin_x:terminus_x] = img_array[:,:,1]

# # plt.imshow(canvas,cmap='gray')
# # plt.show()

# ## compute phase correlation of 1 and 2
# pc = phase_correlation(img_array[:,:,2],img_array[:,:,1])

# ## find the index of the max
# index = np.unravel_index(np.argmax(pc, axis=None), pc.shape)
# print(index)
# ## reset to new origin at top of image 1
# center_y = origin_y
# center_x = origin_x

# ## compute new origin coords
# origin_y = center_y - max_rows + index[0]
# origin_x = center_x + index[1]
# terminus_y = origin_y + max_rows
# terminus_x = origin_x + max_cols

# canvas[origin_y:terminus_y,origin_x:terminus_x] = img_array[:,:,2]

# plt.imshow(canvas,cmap='gray')
# plt.show()

image_center_y = max_rows/2
image_center_x = max_cols/2

images_plotted = 0
last_image_plotted = 0
i = 0
direction = 1

while (images_plotted < num_images):
    ## check to see how many images have been plotted
    if (images_plotted == 0):
        ## then this is the first time through the loop
        ## lay out the coordinates of the origin and terminus of the image
        origin_y = center_y
        origin_x = center_x
        terminus_y = center_y + max_rows
        terminus_x = center_x + max_cols
        
        ## set the corresponding location in the canvas equal to the image
        canvas[origin_y:terminus_y, origin_x:terminus_x] = img_array[:,:,0]

        ## update the images plotted counter and the last image plotted
        images_plotted += 1
        # i += 1
        last_image_plotted = 0

    else:
        ## this is not the first time through the loop
        ## compute the phase correlation of last image plotted and next image in array
        phase_corr = phase_correlation(img_array[:,:,i],img_array[:,:,last_image_plotted])

        ## find the maximum, compare it to the average
        max_phase_correlation = np.amax(phase_corr)
        avg_phase_correlation = np.mean(phase_corr)
        ratio = avg_phase_correlation/max_phase_correlation
        # print(ratio)

        ## if the ratio is less than the following tuned parameter, there is overlap
        if (ratio < 6.2e-5):
            ## find the coordinates of the maximum phase correlation
            max_coordinates = np.unravel_index(np.argmax(phase_corr, axis=None), phase_corr.shape)
            # print("max coordinates: ",max_coordinates)
            ## what quadrant are they in?
            if ((max_coordinates[0] < image_center_y) and (max_coordinates[1] > image_center_x)):
                ## we are in first quadrant
                # print("Q1")
                origin_y = center_y + max_coordinates[0]
                origin_x = center_x + max_coordinates[1] #+ max_cols
                terminus_y = origin_y + max_rows
                terminus_x = origin_x + max_cols

            elif ((max_coordinates[0] < image_center_y) and (max_coordinates[1] < image_center_x)):
                ## we are in second quadrant
                # print("Q2")
                origin_y = center_y - max_rows + max_coordinates[0]
                origin_x = center_x + max_coordinates[1]
                terminus_y = origin_y + max_rows
                terminus_x = origin_x + max_cols

            elif ((max_coordinates[0] > image_center_y) and (max_coordinates[1] < image_center_x)):
                ## we are in third quadrant
                print("Q3")
                print(max_coordinates,"\n")
                origin_y = center_y + max_coordinates[0]
                origin_x = center_x + max_coordinates[1] 
                terminus_y = origin_y + max_rows
                terminus_x = origin_x + max_cols
                
            elif ((max_coordinates[0] > image_center_y) and (max_coordinates[1] > image_center_x)):
                ## we are in fourth quadrant
                # print("Q4")
                # print("max coordinates: ",max_coordinates)
                origin_y = center_y + max_coordinates[0] # + max_cols
                origin_x = center_x + max_cols - max_coordinates[1]
                terminus_y = origin_y + max_rows
                terminus_x = origin_x + max_cols
            
            ## place the image
            canvas[origin_y:terminus_y,origin_x:terminus_x] = img_array[:,:,i]
            
            ## update witness marks
            center_y = origin_y
            center_x = origin_x
            last_image_plotted = i
            images_plotted += 1
            
    ## if we have gone through all images and no overlap, reverse iteration direction
    if (i == (num_images - 1)):
        direction = -1
    elif (i == 0):
        direction = 1
    i += direction
    
plt.imshow(canvas,cmap='gray')
plt.savefig("cell_complete.png")
plt.show()

