#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:37:54 2023

@author: mrittikad
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy import ndimage
from scipy.fftpack import fft2, ifft2, fftshift
from skimage.measure import regionprops
import cv2

def backfinding(image):
    # image = np.round(image, 3)
    img_array = np.array(image, copy=True)                  # writable copy of the input image
    bg_value = img_array[0, 0]                              # find pixel value of background
    img_array[img_array == bg_value] = 0 
    # img_array = img_array.astype(np.uint8)                # Convert to integer values
    labeled_image, num_labels = ndimage.label(img_array)    # find and label foreground and background
    s = regionprops(labeled_image)                          # retrieve properties of the labeled regions
    blobs = np.array([props.area for props in s])           # list of blob areas, largest should be background
#    ind = np.where(blobs == blobs[0])[0]                   # to find other blobs with the same are as the background
#    pix = np.concatenate([s[i].coords for i in ind])
    
    pix = s[np.argmax(blobs)].coords
    
    backmask = coo_matrix((np.ones(pix.shape[0]), (pix[:, 0], pix[:, 1])),
                          shape=img_array.shape, dtype=bool).toarray()

    return backmask

def getaperture(allimages):
    sumImage = backfinding(allimages[0])                    # Initialize with the first image
    for im in range(1, len(allimages)):                     # Start from the second image
        sumImage += backfinding(allimages[im])
    
    meanImage = sumImage / len(allimages)
    aperture = backfinding(meanImage)
    return aperture

def create_checkerboard(width, height, square_size):
    rows = np.ceil(height / square_size).astype(int)
    cols = np.ceil(width / square_size).astype(int)
    # Initialize a checkerboard pattern with zeros
    checkerboard = np.zeros((rows, cols), dtype=int)
    # Fill the checkerboard pattern with alternating white (255) and black (0) squares
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = 255  # White
            else:
                checkerboard[i, j] = 0    # Black
    
    # Repeat the pattern to the desired size
    checkerboard_image = np.kron(checkerboard, np.ones((square_size, square_size), dtype=int))
    checkerboard_image = checkerboard_image[:height, :width]    
    return checkerboard_image

def overlay_aperture(center_x, center_y, avg_face): # how big to present the checkers
    avg_face_aperture = backfinding(avg_face)
    apMask = avg_face_aperture.astype(np.uint8)
    apkernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size as needed
    apMask_smooth = cv2.dilate(apMask, apkernel, iterations=1)
    avg_aperture = apMask_smooth.astype(bool)
    aperture_height, aperture_width = avg_aperture.shape

    # to place the average face aperture on the checkerboard
    start_y, start_x = center_y - aperture_height // 2, center_x - aperture_width // 2
    end_y, end_x = start_y + aperture_height, start_x + aperture_width
    
    return avg_aperture, (start_y, end_y), (start_x, end_x)
    

def phaseScrambleImage(inputImage):
    inputImage = np.array(inputImage)                       # convert to array if not already in array
    inFourier = np.fft.fft2(inputImage)
    inAmp = np.abs(inFourier)
    outPhase = np.angle(np.fft.fft2(np.random.randn(*inputImage.shape)))
    scrambledImage = np.fft.ifft2(inAmp * np.exp(1j * outPhase)).real
    return scrambledImage

# to filter images in each orientation filter
def DoLogGabor(image, FreqPeak, FreqSigma, theta, ThetaSigma, graphics=False):
    n, m = image.shape
    x, y = np.meshgrid(np.arange(m) - m // 2, np.arange(n) - n // 2)  # Creating a grid of coordinates
    polar_grid = x + y * 1j
    center_dist = np.abs(polar_grid)
    center_ang = np.angle(polar_grid)
    
    filters = []    
    # calculate angular spread to create the filter
    ds = np.sin(center_ang) * np.cos(theta) - np.cos(center_ang) * np.sin(theta)    # angular difference between the orientation of the filter 
                                                                                        # and the orientation of each pixel in the image
    dc = np.cos(center_ang) * np.cos(theta) + np.sin(center_ang) * np.sin(theta)    # same as above, finds alignment of pixel with filter
    dtheta = np.arctan2(ds, dc)                                                     # how much the orientation of the filter needs to be rotated to align with the orientation of each pixel
    ang_spread = np.exp(-0.5 * (dtheta / ThetaSigma)**2)                            # angular spread for the filter
            
    # calculate the frequency spread to create the filter
    freq_sd_absolute = (1 / (2**FreqSigma))                                         # frequency spread of the filter      
    rfo = (FreqPeak / min(m, n)) / 0.5                                              # relative frequency offset (rfo) based on the peak frequency of the filter
    sf_spread = np.exp(-0.5 * (np.log(center_dist / rfo) / np.log(freq_sd_absolute))**2)    # spread of the filter's response in SF domain
    sf_spread[n//2, m//2] = 0                                                       # center of SF response set to zero (to avoid artifacts)
            
    # create the filter with both angular and sf spread
    filter_ = sf_spread * ang_spread                                                #  final Log-Gabor filter for a specific orientation and frequency.
    filters.append(filter_)
    filtered_image = np.fft.ifftshift(filter_) * np.fft.fft2(image)
    filtered_image = np.fft.ifft2(filtered_image)
    
    filters = np.stack(filters, axis=-1)                                                 
         
    # if graphics:
        # Plotting code can be added here to visualize the filters and results
    return filtered_image

def generateBlurMask(image, aperture, inner_margin):
    aperture_mask_uint8 = aperture.astype(np.uint8) * 255
    
    # Erode the mask to create a margin inside the mask
    erosion_size = inner_margin  # Number of pixels inside the mask to erode
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(aperture_mask_uint8, kernel, iterations=1)
    
    # Create a smooth gradient for blending
    smooth_gradient = np.zeros_like(eroded_mask, dtype=np.float32)
    smooth_gradient[eroded_mask == 0] = 0  # outside the aperture : background
    smooth_gradient[eroded_mask == 255] = 1  # inside the aperture : face
    
    # reverse to create the blur going outward from the face aperture towards background
    blurred_gradient = cv2.GaussianBlur(1-smooth_gradient, (21, 21), 0)
    
    # Create a blurred version of the orinal face image
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    
    # finally reconstructing the image with aperture blurred
    blended_image = (1-blurred_gradient)*image + blurred_gradient*blurred_image
    # plt.imshow(blended_image, cmap = 'gray')
    
    return blended_image

#### to rename starting images if required ####
# import os
# from PIL import Image
# import cv2
# import numpy as np

# basefolder = '/home/mrittikad/Documents/Projects/orientation_decoding_3T/Stimuli/'
# imagefolder = f'{basefolder}raw_images/unfamiliarOGnames/'
# saveFolder = f'{basefolder}raw_images/unfamiliar'
# imNames = {'image': [],'original ID': [], 'new ID': []}

# for numIm, imfile in enumerate(os.listdir(imagefolder)):
#     if imfile.endswith('.bmp'):
#         imNames['original ID'].append(imfile.split('_')[0])
#         if numIm < 9:
#             imNames['new ID'].append(f'ID0{numIm+1}')
#         else:
#             imNames['new ID'].append(f'ID{numIm+1}')
#         newImName = f"{imNames['new ID'][numIm]}.bmp"        
#         img = Image.open(f'{imagefolder}{imfile}')
#         imNames['image'].append(img)
#         img.save(f'{saveFolder}/{newImName}', 'BMP') 

### to rename starting images into male and female separately ####

# saveFolder = f'{basefolder}raw_images/unfamiliar_gender'
# male_imNames = {'image': [],'original ID': [], 'newID': []}
# female_imNames = {'image': [],'original ID': [], 'newID': []}


# for numIm, imfile in enumerate(os.listdir(imagefolder)):
#     if imfile.endswith('.bmp'):
#         img_id = imfile.split('_')[0]
#         img = Image.open(f'{imagefolder}{imfile}')
#         if img_id.startswith('f'):
#             female_imNames['image'].append(img)
#             female_imNames['original ID'].append(img_id)
#         else:
#             male_imNames['image'].append(img)
#             male_imNames['original ID'].append(img_id)
            
            
# for numIm in range(len(female_imNames['original ID'])):
#     female_imNames['newID'].append(f'F{numIm}')
#     newImName = f"{female_imNames['newID'][numIm]}.bmp"
#     female_imNames['image'][numIm].save(f'{saveFolder}/{newImName}', 'BMP')
    
# for numIm in range(len(male_imNames['original ID'])):
#     male_imNames['newID'].append(f'M{numIm}')
#     newImName = f"{male_imNames['newID'][numIm]}.bmp"
#     male_imNames['image'][numIm].save(f'{saveFolder}/{newImName}', 'BMP')