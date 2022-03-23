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

_LOGGER = logging.getLogger(__name__)

def vizu(binary_yellow_green_mask, mask_pth, pth_img_rgb, pathstudy, config): 

    rgb_binary_mask = np.empty((binary_yellow_green_mask.shape[0], binary_yellow_green_mask.shape[1], 3))

        ### Custom color map ###

    rgb_binary_mask[binary_yellow_green_mask == 1] = (0,100,0)
    rgb_binary_mask[binary_yellow_green_mask == 0] = (255,215,0) 

    label = cv2.imread(mask_pth, 0)
    black_background = np.zeros((binary_yellow_green_mask.shape[0], binary_yellow_green_mask.shape[1]), dtype=np.uint8)

    rgb_binary_mask_2 = rgb_binary_mask.copy()
    cv2.bitwise_not(rgb_binary_mask, rgb_binary_mask_2) 
    result = cv2.bitwise_not(rgb_binary_mask_2, black_background, mask = label)[...,::-1] 
    
    if config['model_parameters']['contrasted'] == 1 :

        contr_result, alpha, beta = automatic_contrast(cv2.imread(pth_img_rgb))

        dst = cv2.addWeighted(contr_result, 0.25, np.array(result).astype(np.uint8), 0.75, 1)
        im_hstacked = cv2.hconcat([contr_result, np.array(dst).astype(np.uint8)])

    else : 
        dst = cv2.addWeighted(cv2.imread(pth_img_rgb), 0.25, np.array(result).astype(np.uint8), 0.75, 1)
        im_hstacked = cv2.hconcat([cv2.imread(pth_img_rgb), np.array(dst).astype(np.uint8)])
      
    chemin = "/" + config['relative_paths']['visualisation'] + "/"

    if not os.path.exists(f'{str(pathstudy)}{chemin}'):
        os.mkdir(f'{str(pathstudy)}{chemin}')

    cv2.imwrite(f'{str(pathstudy)}{chemin}' + str(mask_pth.split('.')[-2].split('/')[-1]) + '.jpg', im_hstacked)



def SegVeg_second_stage(pth_img_rgb, config):

    loaded_model = pickle.load(open(config['model_parameters']['model'], 'rb')) # load the SegVeg model

            ### If RGB image luminosity/contrast enhancement is ON ###
    
    if config['model_parameters']['contrasted'] == 1 :

        arr_image = cv2.imread(pth_img_rgb)

        contr_result, alpha, beta = automatic_contrast(arr_image)
        arr_image = contr_result
        arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(arr_image)

    else :

        im = Image.open(pth_img_rgb)

        
    width, height = im.size 
    # Compute the pixel features for the all pixels RGB images
    hsv = np.array(im.convert(mode='HSV'))
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(im, rgb2lab))
    cmyk = np.array(im.convert(mode='CMYK'))
    ycbcr = np.array(im.convert(mode='YCbCr'))
    rgb_lab = io.imread(pth_img_rgb)
    Labb = color.rgb2lab(rgb_lab)
    rgb = np.array(im)
    r = rgb[:,:,0].flatten('F')
    g = rgb[:,:,1].flatten('F')
    b = rgb[:,:,2].flatten('F')

    h = ((hsv[:,:,0].flatten('F')) * 360) / 255
    s = (hsv[:,:,1].flatten('F')) / 2.55
    a = Labb[:,:,1].flatten('F')
    bb = Lab[:,:,2].flatten('F')
    ge =  np.mean([r,g,b], axis = 0)
    
    CMYlist = [1 - r / 255, 1 - g / 255, 1 - b / 255] 
    CMYlist = np.array([min(idx) for idx in zip(*CMYlist)])
    m = ((1 - g / 255 - CMYlist ) / (1 - CMYlist )) * 100
    ye = ((1 - b / 255 - CMYlist ) / (1 - CMYlist )) * 100
    
    cb = ycbcr[:,:,1].flatten('F')
    cr = ycbcr[:,:,2].flatten('F')


    def rgb2yiq(r, g, b):
    
        y1y = (0.299 * r + 0.587 * g + 0.114 * b)
        y1i = (0.596 * r - 0.275 * g - 0.321 * b)
        y1q = (0.212 * r - 0.523 * g + 0.311 * b)

        return y1y, y1i, y1q

    i = rgb2yiq(r, g, b)[1]
    q = rgb2yiq(r, g, b)[2]
    

    X = np.column_stack((r,g,b,h,s,a,bb,ge,m,ye,cb,cr,i,q))
    X = np.nan_to_num(X) # Handle black pixels (avoiding Input contains NaN, infinity or a value too large for dtype('float64') error)
    

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
    
    thresh = config['model_parameters']['thresh']
    y_pred = (loaded_model.predict_proba(X)[:,1] >= thresh).astype(int)
    
    segmented = y_pred.reshape((width, height))
    segmented = cv2.flip(segmented, 1)
    segmented = cv2.rotate(segmented, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return segmented


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
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    contr_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (contr_result, alpha, beta)



def yellowclassif(pathstudy, config):     # First called function in cli.py file | It loops over the images to return Visualization (only for 1 image per Plot)
                                          # and Remplace/Ecrase Vegetation/Background masks in config.json/'vegetation_masks' into Green vegetation/Senescence/Background for all images in Plots

    with open(config) as json_file:       # Import .json file for parameters and paths bash
        config = json.load(json_file)
    
    list_paths_masks_per_session = glob.glob(str(pathstudy) + '/' + config['relative_paths']['vegetation_masks'] + "/*.png") # List of all plant segmentation masks per Session | Full paths

    list_all_plots_per_session = list(set([os.path.basename(t).split('_Camera')[0] for t in list_paths_masks_per_session])) # List all the plots per Session | Name of Plots, not full paths, ex : [Plot204, Plot205]

    if not list_all_plots_per_session:
        _LOGGER.error('No images found in this Session : {}'.format(pathstudy))

    list_plotId = []

    for plot_name in tqdm(list_all_plots_per_session, desc = "Compute Green and Yellow segmentation : Loop for {} microplots :".format(len(list_all_plots_per_session))): # Loop over the plots

        list_masks_plot_paths = [i for i in list_paths_masks_per_session  if (plot_name in os.path.basename(i))] # List all the plant segmentation masks per plot per Session | Full paths

        for mask_pth in tqdm(list_masks_plot_paths, desc = "Considered microplot : {} with {} image(s)".format(plot_name, len(list_masks_plot_paths))): # One plant segmentation mask of the considered plot | Full path
            
            mask = np.array(Image.open(mask_pth)) # Loaded mask 
            pth_img_rgb = mask_pth.replace(config['relative_paths']['vegetation_masks'], config['relative_paths']['rgb_images']).replace(".png", ".jpg") # Path of the RGB image
            img_rgb = np.array(Image.open(pth_img_rgb)) # Loaded RGB image
            binary_yellow_green_mask = SegVeg_second_stage(pth_img_rgb, config) # SegVeg_second_stage model | Full image binary yellow and green segmentation 
                                                                                # (including Background from Vegetation mask, work on Back. pixels will be done after) : 0 stands for Senescent pixels, 1 for Green

            mask[(mask == 1) & (binary_yellow_green_mask != 1)] = 2  # Back. work mentionned above : 2 stands for Background pixels
            img = Image.fromarray(mask.astype(np.uint8))
            img.convert('L').save(mask_pth) # Ecrase Vegetation/Background binary mask and save Green vegetation/Senescence/Background binary mask



            plotId = os.path.basename(mask_pth).split('_Camera')[0]

            if plotId not in list_plotId: # Save one visualization image per plot, look at first image loaded by ImageLoader glob + set + tdqm, add ID plot (plot_name) in list, if exists do not save
                list_plotId.append(plotId)                                                              
                vizu(binary_yellow_green_mask, mask_pth, pth_img_rgb, pathstudy, config) #Vizu, horizontal stack + overlay transparency


            


                
