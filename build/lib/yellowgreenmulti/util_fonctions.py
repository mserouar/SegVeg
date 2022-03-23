import csv
import numpy as np
import colorsys
import pandas as pd
import os
import math
import pickle
import pylab as plt
import cv2
import time
import progressbar
import tqdm
import PIL
import warnings
import timeit
import scikitplot as skplt
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors

from glob import glob
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from tqdm import tqdm
from p_tqdm import p_map
from PIL import Image 


def rgb2cmyk(r, g, b):

    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100

    C = 1 - r / 255
    M = 1 - g / 255
    Y = 1 - b / 255

    min_cmy = min(C, M, Y)

    C = (C - min_cmy) / (1 - min_cmy)
    M = (M - min_cmy) / (1 - min_cmy)
    Y = (Y - min_cmy) / (1 - min_cmy)
    K = min_cmy

    return C * 100, M * 100, Y * 100, K * 100

def rgb2greyscale(r, g, b):
    
    ge = (r + g + b) / 3

    return ge

def rgb2ycbcr(r, g, b):
    
    Y =   16 +  65.738 * r / 256 + 129.057 * g / 256 +  25.064 * b / 256
    Cb = 128 -  37.945 * r / 256 -  74.494 * g / 256 + 112.439 * b / 256
    Cr = 128 + 112.439 * r / 256 -  94.154 * g / 256 -  18.285 * b / 256

    return Y, Cb, Cr

def rgb2yiq(r, g, b):
    
    y1y = (0.299 * r + 0.587 * g + 0.114 * b)
    y1i = (0.596 * r - 0.275 * g - 0.321 * b)
    y1q = (0.212 * r - 0.523 * g + 0.311 * b)

    return y1y, y1i, y1q


def rgb2hsv(r, g, b):

    r, g, b = r / 255, g / 255,  b /255

    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn

    if mx == mn:
        H = 0
    elif mx == r:
        H = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        H = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        H = (60 * ((r - g) / df) + 240) % 360

    if mx == 0:
        S = 0
    else:
        S = (df / mx) * 100

    V = mx * 100

    return H, S, V

def rgb2lab (r,g,b) : #r,g,b inputColor

   num = 0
   RGB = [0, 0, 0]
   inputColor = r,g,b

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** (2.4)
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505

   XYZ[0] = round(X, 4)
   XYZ[1] = round(Y, 4)
   XYZ[2] = round(Z, 4)

   XYZ[0] = float(XYZ[0]) / 95.047         
   XYZ[1] = float(XYZ[1]) / 100.0         
   XYZ[2] = float(XYZ[2]) / 108.883      

   num = 0

   for value in XYZ :

       if value > 0.008856 :
           value = value ** (1/3)
       else :
           value = (7.787 * value ) + (16 / 116)

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = (116 * XYZ[1]) - 16
   a = 500 * (XYZ[0] - XYZ[1])
   b = 200 * (XYZ[1] - XYZ[2])

   Lab [0] = round(L, 4)
   Lab [1] = round(a, 4)
   Lab [2] = round(b, 4)

   return Lab



