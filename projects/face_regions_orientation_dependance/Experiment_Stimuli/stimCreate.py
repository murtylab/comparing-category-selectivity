#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:46:38 2023

@author: mrittikad
"""
#%% =============================================================================
# 1. paths and imports
# basefolder = '/Users/mrittikadey/Documents/Projects/orientation_decoding_3T/Stimuli/'
basefolder = 'C:/Users/Adminuser/OneDrive/Documents/Projects/orientation_decoding_3T/Stimuli/'
# basefolder = '/home/mrittikad/Documents/Projects/orientation_decoding_3T/Stimuli/'
imagefolder = f'{basefolder}raw_images/unfamiliar_gender/'
maskfolder = 'C:/Users/Adminuser/OneDrive/Documents/Projects/orientation_decoding_3T/Stimuli/Masks/'
resizedStim = f'{basefolder}raw_images/resized/'

checkersFolder =  f'{basefolder}Checkers/'

outpath_contrast = f'{basefolder}stimuliFinal/blurred/' 
outpath_identity = f'{basefolder}stimuliFinal_identity/blurred/' 
outpath_illustration = f'{basefolder}stimuliFinal/blurred_and_masked/' 

from PIL import Image
import os
os.chdir(basefolder)
os.makedirs(outpath_contrast, exist_ok=True)
os.makedirs(outpath_identity, exist_ok=True)
os.makedirs(outpath_illustration, exist_ok=True)
import sys
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import copy
import glob
from stimCreate_functions import *

#%% =============================================================================
# 2. Loading in images
# List image files in the imagefolder directory
nim = [imfile for imfile in os.listdir(imagefolder) if imfile.endswith('.bmp')]

# Prepare a list to store the loaded images and store different image versions
imset = {'raw_stim': [], 'resized': [], 'aperture': [], 'aperture_inverted': [],
         'norm_stim': [], 'eq_stim': [], 'scrambled_stim': [], 'with_scr_back': []}

imset['coords'] = {'facepix': [], 'backpix': []}

filtered_stim = {'Intact': {}, 'Inverted':{},
                 'Negated': {}, 'Scrambled': {}}

im2avg_stim =  [None] * len(nim)
# Load the images (pixel value between 0 - 255)
for theim in range(len(nim)):
    im_path = f'{imagefolder}{nim[theim]}'
    im_stim = Image.open(im_path).convert('L')          # Load image and convert to grayscale by 'L'
    imset['raw_stim'].append(im_stim)

# paddim = max(xySize_raw) - min(xySize_raw)
# xySize_stim = [max(xySize_raw), max(xySize_raw)]
# paddims = (desired_size[0] - xySize_stim[0]) + (max(xySize_raw) - min(xySize_raw))

#%% =============================================================================
# 3. Resizing and padding images
desired_size = (550, 550)       # has to be square
scaleImage = 3                  # scaling padded image to 3 times desired_size (for proper resizing)
scaleSize = (desired_size[0]*scaleImage, desired_size[1]*scaleImage) # now 1650,1650

# Get raw image size
xySize_raw = imset['raw_stim'][0].size                  # Assuming all raw images have the same size

# Calculate padding needed to make images square
pad_width = (round((scaleSize[0]-xySize_raw[0])/2), round((scaleSize[0]-xySize_raw[0])/2))
pad_height = (round((scaleSize[1]-xySize_raw[1])/2), round((scaleSize[1]-xySize_raw[1])/2))

# Loop through images to find aperture, resize and pad them
for theim in range(len(nim)):
    im_stimb = imset['raw_stim'][theim]
    # pad the images
    im_stimb = np.pad(im_stimb, [pad_height,pad_width], mode='edge') # edge is equivalent to 'replicate' in MATLAB
   
    # Resize images to desired_size (Convert the numpy array back to a PIL Image for resizing)
    im_stimb = (Image.fromarray(im_stimb)).resize(desired_size, Image.LANCZOS)    
    imset['resized'].append(im_stimb)
    im_stimb.save(f'{resizedStim}{nim[theim]}', 'BMP')
    
    # get image aperture
    back_mask = backfinding(im_stimb)
    # smooth the resulting mask
    # convert boolean values to integers (0 for False, 1 for True)
    apMask = back_mask.astype(np.uint8)
    apkernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size as needed
    apMask_smooth = cv2.dilate(apMask, apkernel, iterations=1)
    aperture = apMask_smooth.astype(bool)
    imset['aperture'].append(aperture) # append final aperture per face
    
    # save coordinates of the face
    facepixindex = np.argwhere(aperture == True)
    facepix_coords = (facepixindex[:, 0], facepixindex[:, 1])           
    imset['coords']['facepix'].append(facepix_coords)
    
    # save coordinates of the background
    backpixindex = np.argwhere(aperture == False)
    backpix_coords = (backpixindex[:, 0], backpixindex[:, 1])           
    imset['coords']['backpix'].append(backpix_coords)
    im2avg_stim[theim] = im_stimb
    # Display the image (optional)
    # im_stimb.show()
    
    
#%% =============================================================================
# Resizing the mask
#load a mask of the face
alphamask = Image.open(f'{maskfolder}alphamask.bmp')
alphamask = np.pad(alphamask, [20,20], mode='edge') # adjust padding as needed ([20,20] works fine for me)
alphamask = (Image.fromarray(alphamask)).resize(desired_size, Image.LANCZOS)
alphamask = np.array(alphamask)
saveFileName = f'{maskfolder}small_alphamask.bmp'
cv2.imwrite(saveFileName, alphamask)

    
#%% =============================================================================
# 4. Normalising + Equalising across images
LC = [0.45, 0.05]                # desired luminance and contrast
#totalPC = 0
faceWidth = 0
for theim in range(len(nim)):
    # Normalize the images
    im_stimb = np.array(imset['resized'][theim]) / 255.0                    # Convert image to array and scale pixel values between 0 - 1
    im2avg_stim[theim] = np.array(im_stimb)
    saveIm = copy.deepcopy(im_stimb) 
    facepix = copy.deepcopy(im_stimb[imset['coords']['facepix'][theim]])    # select face pixels
    
    # change the range to avoid clipping further down
    numClipping = np.sum((facepix < 0.1) | (facepix > 0.9))
    percentClipping = (numClipping/facepix.size)*100                        # percentage clipping for each image
    #totalPC +=  percentClipping
    rescaledfacepix = exposure.rescale_intensity(facepix, in_range= (np.min(facepix), np.max(facepix)), out_range=(0.1, 0.9))
    
    # normalize face pixel values
    normfacepix = (rescaledfacepix - np.mean(rescaledfacepix)) / np.std(rescaledfacepix)        
    saveIm[imset['coords']['facepix'][theim]] = normfacepix
    imset['norm_stim'].append(saveIm)                                       # pixel values here are scaled between 0 - 1
    print(f'{theim} norm check - mean: {abs(np.mean(normfacepix)):.2f} - std: {abs(np.std(normfacepix)):.2f}, clipping: {percentClipping:.2f}%')
    
    # equalise the image (Set desired luminance and contrast) 
    eqfacepix = (normfacepix * LC[1]) + LC[0]
    saveIm[imset['coords']['facepix'][theim]] = eqfacepix                   # replace face pixels of the original image by the equalised ones      
    
    # Set desired luminance and contrast for background
    saveIm[imset['coords']['backpix'][theim]] = LC[0]
    imset['eq_stim'].append(saveIm)
    faceWidth += np.max(np.sum(imset['aperture'][theim], axis=1)) # in pixels

# count width of facepixels
avg_facewidth = round(faceWidth/len(nim)) # in pixels
#avgPC = totalPC/len(nim)
# Convert and save the average face image    
avgface = np.mean(im2avg_stim, axis=0)
plt.imshow(avgface, cmap='gray')
avgface_img = Image.fromarray((avgface * 255).astype(np.uint8))     # Convert to image format
avgface_img.save(f'{basefolder}avgface_set.bmp', 'BMP')             # Save the average face image as a BMP file

#%% =============================================================================
# 5. Create fullscreen checkerboard stimuli
final_checkerboards = {'checkerBack': [], 'checkerFace': []}
listCheckers = glob.glob(f'{checkersFolder}/*.bmp')
if not listCheckers:
    screen_width, screen_height, square_size = 1920, 1080, 100 # for BOLD screen
    checkerboard_array = create_checkerboard(screen_width, screen_height, square_size)
    center_x, center_y= screen_width // 2, screen_height // 2 # find center of the big checkerboard
    avg_aperture, face_y_coords, face_x_coords = overlay_aperture(center_x, center_y, avgface)
    
    # background checkers
    # where avg_aperture == True, put 127 (grey) on face
    checkerboard_array[face_y_coords[0]:face_y_coords[1], face_x_coords[0]:face_x_coords[1]] = np.where(avg_aperture, 127, checkerboard_array[face_y_coords[0]:face_y_coords[1], face_x_coords[0]:face_x_coords[1]])
    checkerboard_array_inv = 255 - checkerboard_array
    final_checkerboards['checkerBack'].append((checkerboard_array, checkerboard_array_inv))
    
    # face checkers
    square_size = 50 # smaller squares for face checkers 
    checkerboard_array = create_checkerboard(screen_width, screen_height, square_size) # create a new checkerboard array
    grey_background = np.full((screen_height, screen_width), 127, dtype=np.uint8)
    # where avg_aperture == True, put checkerboards on face
    grey_background[face_y_coords[0]:face_y_coords[1], face_x_coords[0]:face_x_coords[1]] = np.where(avg_aperture, checkerboard_array[face_y_coords[0]:face_y_coords[1], face_x_coords[0]:face_x_coords[1]], 127)
    grey_background_inv = 255 - grey_background
    final_checkerboards['checkerFace'].append((grey_background, grey_background_inv))
    
    for checkName in final_checkerboards.keys():
        saveIm = Image.fromarray(final_checkerboards[checkName][0][0].astype(np.uint8))
        saveIm.save(f'{checkersFolder}{checkName}_ori.bmp', 'BMP')
        saveIm_inv = Image.fromarray(final_checkerboards[checkName][0][1].astype(np.uint8))
        saveIm_inv.save(f'{checkersFolder}{checkName}_inv.bmp', 'BMP')    

#%% =============================================================================
# 6. Iterative scrambling 
# we do this so that scrambled stimuli generated later are not too contaminated by the uniform background
fig_over_back_ratio = len(facepix) / np.prod(desired_size)          # Calculate the proportion of face versus background pixels of one example face
niter = 500 # 500                                                    # no. of interations - several 100s
save_ScrFolder = basefolder + 'scrambled_background/'
os.makedirs(save_ScrFolder, exist_ok = True)

for theim in range(len(nim)):
    current_image = f'scr_{theim}.bmp'
    # select face and background pixels for this particular image
    facepix_coords = imset['coords']['facepix'][theim]
    backpix_coords = imset['coords']['backpix'][theim]
    
    if current_image not in os.listdir(save_ScrFolder):
        scrimage = phaseScrambleImage(imset['norm_stim'][theim])
        combiface = scrimage.copy()
        combiface[facepix_coords] = imset['norm_stim'][theim][facepix_coords]
        combiface[backpix_coords] = scrimage[backpix_coords]            # scrambled image
        
        # normalizing the image
        combiface = (combiface - np.mean(combiface)) / np.std(combiface)
        print(f'{theim} iterative scrambling - mean: {np.mean(combiface):.2f} - std: {np.std(combiface):.2f}')
        
        iterscrface_stim = [None] * niter                               # preallocate space for the scrabling iterations
        for theiter in range(niter):
            if theiter == 0:
                interim = combiface
            else:
                interim = iterscrface_stim[theiter - 1]
                
            scrimage = phaseScrambleImage(interim)
            scrimage[facepix_coords] = imset['norm_stim'][theim][facepix_coords] # paste face image back on scrambled image
            scrimage = (scrimage - np.mean(scrimage)) / np.std(scrimage) # normalise scrambled image
            iterscrface_stim[theiter] = scrimage
            # plt.imshow(iterscrface_stim[theiter], cmap='gray')
        
        scrimage = phaseScrambleImage(iterscrface_stim[theiter])
        scrimage = (scrimage - np.min(scrimage)) / (np.max(scrimage) - np.min(scrimage)) # scale again to keep pixels between 0 and 1
        scrimage = (scrimage - np.mean(scrimage)) / np.std(scrimage)    # final normalization
        scrimage = (scrimage * LC[1]) + LC[0]
        imset['scrambled_stim'].append(scrimage)
        save_Scr_image = Image.fromarray((scrimage * 255).astype(np.uint8))     # Convert to image format
        save_Scr_image.save(f'{save_ScrFolder}scr_{theim}.bmp', 'BMP')          # Save the average face image as a BMP file
    else:
        scrimage =  Image.open(f'{save_ScrFolder}scr_{theim}.bmp')
        imset['scrambled_stim'].append(scrimage)
        scrimage_arr = np.array(copy.deepcopy(scrimage))
        scrback = scrimage_arr/255.0
        scrback[facepix_coords] = imset['norm_stim'][theim][facepix_coords]
        imset['with_scr_back'].append(scrback)

#%% =============================================================================
# 7. Orientation Filtering 
# Filters
FreqPeak = 100
FreqSigma = 1000  # bandwidth (in octaves)
Angle = np.arange(0, 180, 30)  # don't do 180
thetaAngles = list(np.radians(Angle))
thetasig = 15
ThetaSigma = np.radians(thetasig)  # bandwidth in radians
final_LC = [0.45, 0.035] # final luminance to adjust 
final_LC_illustration = [0.45, 0.05] # final luminance to adjust 

# for Natural/Intact images
for theta_idx, theta in enumerate(thetaAngles):
    stim_type = ['Int', 'Inv', 'Scr'] #'Neg']
    
    filtered_stim['Intact'][theta_idx] = []
    filtered_stim['Inverted'][theta_idx] = []
    filtered_stim['Scrambled'][theta_idx] = []
    
    for theim in range(len(imset['with_scr_back'])):
        for stim in stim_type:
            daim = imset['with_scr_back'][theim] # starting point for all types of stimulus except scrambled
            
            if stim == 'Scr':
                daim = imset['scrambled_stim'][theim] # starting point for scrambled stimuli
            
            elif stim == 'Neg':
                daim = 1 - daim # contrast negation
            
            daim = (daim - np.mean(daim))/ np.std(daim)
            filtim = DoLogGabor(daim, FreqPeak, FreqSigma, theta, ThetaSigma, graphics=False)
            real1 = np.real(filtim)
        
            # select face and background pixels for this image
            facepix_coords = imset['coords']['facepix'][theim]
            backpix_coords = imset['coords']['backpix'][theim]

            # (normalize +) equalize lum and contrast of the face pixels independently
            facePixFiltered = real1[facepix_coords]
            facePixFiltered = (facePixFiltered - np.mean(facePixFiltered)) / np.std(facePixFiltered)
            facePixFiltered = (facePixFiltered * final_LC_illustration[1]) + final_LC_illustration[0]
            
            # # (normalize +) equalize lum and contrast of the BACKGROUND pixels independently
            # backPixFiltered = real1[backpix_coords]
            # backPixFiltered = (backPixFiltered - np.mean(backPixFiltered)) / np.std(backPixFiltered)
            # backPixFiltered = (backPixFiltered * final_LC[1]) + final_LC[0]
                    
            # reconstruct the image with filtered background and filtered face
            canvas = np.zeros((daim.shape[0], daim.shape[1]))
            canvas[backpix_coords] = final_LC_illustration[0]
            canvas[facepix_coords] = facePixFiltered
            canvas_blurred = generateBlurMask(canvas, imset['aperture'][theim], 10)
            grayImSave = (plt.cm.gray(canvas_blurred)*255).astype(np.uint8)     # save in grayscale

            if stim == 'Int':    
                filtered_stim['Intact'][theta_idx].append(canvas_blurred)
                toMaskImg = copy.deepcopy(canvas_blurred)
            
            elif stim == 'Inv':
                canvas_inv = np.rot90(np.rot90(canvas_blurred)) # rotating the full image by 2 times 90 degrees to INVERT
                filtered_stim['Inverted'][theta_idx].append(canvas_inv)
                grayImSave = (plt.cm.gray(canvas_inv)*255).astype(np.uint8)
                toMaskImg = copy.deepcopy(canvas_inv)

            elif stim == 'Scr':
                filtered_stim['Scrambled'][theta_idx].append(canvas_blurred)
                toMaskImg = copy.deepcopy(canvas_blurred)
                
            # overlay alphamask here 
            overlay_mask = 1 - (alphamask / 255.0)
            final_masked_img = toMaskImg*overlay_mask
            bg = np.full_like(toMaskImg, final_LC_illustration[0])
            blended = toMaskImg*overlay_mask + bg*(alphamask / 255.0)
            maskedImSave = (plt.cm.gray(blended)*255).astype(np.uint8)            
            
            imageID = (os.path.splitext(nim[theim])[0]).split('_')[0]
            saveFileName = f'{imageID}_ori{round(np.rad2deg(theta))}_{stim}.bmp'
            cv2.imwrite(f'{outpath_contrast}/{saveFileName}', grayImSave)
            # cv2.imwrite(f'{outpath_identity}/{saveFileName}', grayImSave)
            cv2.imwrite(f'{outpath_illustration}/{saveFileName}', maskedImSave)
            

# for full spectrum images 
for theim in range(len(imset['with_scr_back'])):
    for stim_type in ['Int', 'Inv', 'Scr']: #'Neg'
    
        daim = imset['with_scr_back'][theim]
        
        if stim_type == 'Scr':
            daim = imset['scrambled_stim'][theim]
       
        elif stim_type == 'Neg':
            daim = 1 - daim
        
        daim = (daim - np.mean(daim))/ np.std(daim)
        
        # select face and background pixels for this image
        facepix_coords = imset['coords']['facepix'][theim]
        backpix_coords = imset['coords']['backpix'][theim]
        
        # (normalize +) equalize lum and contrast of the face pixels independently
        facePixFiltered = daim[facepix_coords]
        facePixFiltered = (facePixFiltered - np.mean(facePixFiltered)) / np.std(facePixFiltered)
        facePixFiltered = (facePixFiltered * final_LC_illustration[1]) + final_LC_illustration[0]
        
        # reconstruct the image with filtered background and filtered face
        canvas = np.zeros((daim.shape[0], daim.shape[1]))
        canvas[backpix_coords] = final_LC_illustration[0]
        canvas[facepix_coords] = facePixFiltered
        canvas_blurred = generateBlurMask(canvas, imset['aperture'][theim], 10)
    
        daimSave = (plt.cm.gray(canvas_blurred)*255).astype(np.uint8)
        
        if stim_type == 'Inv':
            canvas_inv = np.rot90(np.rot90(canvas_blurred)) # rotating the full image by 2 times 90 degrees to INVERT
            daimSave = (plt.cm.gray(canvas_inv)*255).astype(np.uint8)
            toMaskImg = copy.deepcopy(canvas_inv)
        else:
            toMaskImg = copy.deepcopy(canvas_blurred)
        
        # overlay alphamask here too
        overlay_mask = 1 - (alphamask / 255.0)
        final_masked_img = toMaskImg*overlay_mask
        bg = np.full_like(toMaskImg, final_LC_illustration[0])
        blended = toMaskImg*overlay_mask + bg*(alphamask / 255.0)
        maskedImSave = (plt.cm.gray(blended)*255).astype(np.uint8)
                  
        imageID = (os.path.splitext(nim[theim])[0]).split('_')[0]
        saveFileName = f'{imageID}_oriFS_{stim_type}.bmp'
        cv2.imwrite(f'{outpath_contrast}/{saveFileName}', daimSave)
        cv2.imwrite(f'{outpath_identity}/{saveFileName}', daimSave)
        cv2.imwrite(f'{outpath_illustration}/{saveFileName}', maskedImSave)

        