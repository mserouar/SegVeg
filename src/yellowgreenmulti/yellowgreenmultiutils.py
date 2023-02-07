import logging
import pathlib
import pandas as pd
import os
import glob
import numpy as np
import PIL
from PIL import Image, ImageCms
import pickle
from tqdm import tqdm
import cv2
from skimage import color, io
import time
#from joblib import Parallel, delayed
import json
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def get_features(image) :

    pil_image = Image.fromarray(image)

    hsv = np.array(pil_image.convert(mode='HSV'))
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(pil_image, rgb2lab))
    ycbcr = np.array(pil_image.convert(mode='YCbCr'))
    Labb = color.rgb2lab(image)
    r = image[:, :,0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    h = ((hsv[:,:,0]) * 360) / 255
    s = (hsv[:, :, 1]) / 2.55
    a = Labb[:, :, 1]
    bb = Lab[:, :, 2]
    ge =  np.mean([r,g,b], axis = 0)
    
    CMYlist = [1 - r / 255, 1 - g / 255, 1 - b / 255] 
    CMYlist = np.array([np.min(idx) for idx in zip(*CMYlist)])
    m = ((1 - g / 255 - CMYlist ) / (1 - CMYlist )) * 100
    ye = ((1 - b / 255 - CMYlist ) / (1 - CMYlist )) * 100
    
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    i = (0.596 * r - 0.275 * g - 0.321 * b)
    q = (0.212 * r - 0.523 * g + 0.311 * b)
    
    model_input = np.stack((r,g,b,h,s,a,bb,ge,m,ye,cb,cr,i,q), axis=2).squeeze(0)
    model_input = np.nan_to_num(model_input) # Handle black pixels (avoiding Input contains NaN, infinity or a value too large for dtype('float64') error)

    return model_input

    ### Old Parallelise .predict() method ###
    """
    n_cores = config['model_parameters']['n_cores']
    n_samples = X.shape[0]
    slices = [(int(n_samples*i/n_cores), int(n_samples*(i+1)/n_cores)) for i in range(n_cores)]
    data_chunks = [X[i[0]:i[1]] for i in slices]

    if n_cores > 1:
        jobs = (delayed(loaded_model.predict)(array) for array in data_chunks)
        parallel = Parallel(n_jobs=n_cores)

        y_pred  = parallel(jobs)
    else:
        y_pred = [loaded_model.predict(array) for array in data_chunks]

    y_pred  = np.concatenate(y_pred) # Instead of vstack (avoiding all the input array dimensions for the concatenation axis must match exactly error)
    """


def automatic_contrast(image, clip = 0.01):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Cumulative distribution from the histogram
    cumul = []
    cumul.append(float(hist[0]))
    for index in range(1, hist_size):
        cumul.append(cumul[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = cumul[-1]
    clip *= (maximum/100.0)
    clip /= 2.0

    # Left cut
    minimum_gray = 0
    while cumul[minimum_gray] < clip:
        minimum_gray += 1

    # Right cut
    maximum_gray = hist_size -1
    while cumul[maximum_gray] >= (maximum - clip):
        maximum_gray -= 1

    # Alpha and beta values
    alpha = (255 / (maximum_gray - minimum_gray)) + 0.4
    beta = (-minimum_gray * alpha)*2


    # Calculate new histogram with desired range and show histogram 
    contr_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (contr_result, alpha, beta)



def prediction(image, model, threshold, contrasted, mask) :

    # Ff contrasted apply image contrast
    if contrasted == 1 :
        image, _, _ = automatic_contrast(image)
    
    # Revert channels to have RGB /!\ IMPORTANT
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # If Mask doesn't exist create one full of 1's
    if mask is None:
        mask = np.ones((height, width))

    # Flatten arrays
    image = image.reshape((width*height), 1, 3)
    mask = mask.reshape((width*height), 1)

    # Select indices of vegetation pixels
    yellow_green_mask = np.zeros(image.shape[:-1]) # Null array to set bckg pixels to 0
    vegetation_pixels = mask > 0 # Get vegetation pixels indices
    image = image[None, vegetation_pixels]

    # Apply preprocessing to add features to each pixels
    featured_image = get_features(image)
        
    # Predictions on vegetation pixels
    y_pred = (model.predict_proba(featured_image)[:,1] >= threshold).astype(int)

    # Replace pixels of null array at vegetation indices with output of classification
    yellow_green_mask[vegetation_pixels] = y_pred
    # Reshape
    mask[(mask == 1) & (yellow_green_mask != 1)] = 2
    mask = mask.reshape((height, width))
    # Rotate image
    
    return mask


def visualisation(rgb_image: np.ndarray, yg_mask: np.ndarray) -> np.ndarray:
    
    image_copy = rgb_image.copy()
    image_copy[yg_mask==1] = (0,100,0)
    image_copy[yg_mask==2] = (0,215,255)

    visualisation = cv2.addWeighted(rgb_image,0.25, image_copy, 0.75, 0)

    return visualisation



def yellowclassif(pathstudy, config):     # First called function in cli.py file | It loops over the images to return Visualization (only for 1 image per Plot)
                                          # and Remplace/Ecrase Vegetation/Background masks in config.json/'vegetation_masks' into Green vegetation/Senescence/Background for all images in Plots

    with open(config) as json_file:       # Import .json file for parameters and paths bash
        config = json.load(json_file)

    vegetation_masks_folder = config['relative_paths']['vegetation_masks']
    rgb_images_folder = config['relative_paths']['rgb_images']
    threshold =  config['model_parameters']['thresh']
    contrasted = config['model_parameters']['contrasted']
    model_path = config['model_parameters']['model']
    visualisation_folder =  config['relative_paths']['visualisation']


    vegetation_masks_path = Path(pathstudy) / vegetation_masks_folder
    rgb_images_path = Path(pathstudy) / rgb_images_folder
    visualisation_path = Path(pathstudy) / visualisation_folder
    # create visualisation folder
    visualisation_path.mkdir(parents=True, exist_ok=True)
    # load the model
    model = pickle.load(Path(model_path).open("rb")) # load the SegVeg model

    for maskp in tqdm(list(vegetation_masks_path.iterdir()), desc="Compute Green and Yellow segmentation : Loop for {} microplots :".format(len(list(vegetation_masks_path.iterdir())))):
        # get name_img.png
        name = maskp.name 
        # get corresponding rgb path
        rgbp = rgb_images_path / name.replace(".png", ".jpg") #CHange as you wish
        # load image and masks
        image = cv2.imread(rgbp.as_posix(), -1)
        mask = cv2.imread(maskp.as_posix(),-1)
        # mask = mask / 255

        # model prediction output 0,1,2 segmentation mask
        yellow_green_mask = prediction(image, model, mask=mask, threshold=threshold, contrasted=contrasted)
        # create visualisation and visualisation path
        visup = visualisation_path / name.replace(".png", ".jpg")
        visu = visualisation(image, mask)
        # save yellow green mask and visualisation
        cv2.imwrite(maskp.as_posix(), yellow_green_mask)
        cv2.imwrite(visup.as_posix(), visu)
