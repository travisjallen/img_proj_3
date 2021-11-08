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

## read the images
img_0 = io.imread()
