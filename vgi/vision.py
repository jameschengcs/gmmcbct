# VGI - Vision, Graphics, and Imaging 
# Common module
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import vgi.common
import numpy as np
 
import cv2 
import copy
from PIL import Image 
import matplotlib.pyplot as plt
 
import torch
import torch.nn.functional as F
 
import scipy.stats as stats
from scipy import ndimage
from datetime import datetime

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

__all__ = ()
__all__ += ('featureORB', 'loadImgORB', 'getFeatureLevel', 'featureValue', )

# ---------------------------------------------
# Feature detection
# image: input image with integer-pixel format
# desc: Output the description for each keypoint
# return a set of cv::KeyPoint, and a set of descriptions
def featureORB(image, desc = False, toUInt8 = True):
    img = image
    if toUInt8:
        img = vgi.common.toU8(img)
    orb = cv2.ORB_create()
    KP = orb.detect(img, None)
    D = None
    if desc:
        KP, D = orb.compute(img, KP)
    return KP, D


def loadImgORB(path, normalize = True, gray = True, desc = False):
    if gray:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    KP, D = featureORB(img, desc = desc)
        
    if normalize:
        grayscale = 255.0
        if(img.dtype == np.uint16):
            grayscale = 65535.0
        imgData = np.asarray(img) / grayscale
        return imgData, KP, D
    else:
        return img, KP, D     

def getFeatureLevel(keypoints, level, thres = 0.0):
    KPs = []
    for p in keypoints:
        if p.octave == level and p.response >= thres:
            KPs += [p] 
    return KPs

def featureValue(image, keypoints):
    average_value = []
    if image.ndim == 2:
        nchannels = 1
    elif image.ndim > 2:
        nchannels = image.shape[-1]
    for keypoint in keypoints:
        circle_x =      int(keypoint.pt[0])
        circle_y =      int(keypoint.pt[1])
        circle_radius=  int(keypoint.size/2)
        #copypasta from https://stackoverflow.com/a/43170927/2594947
        circle_img = np.zeros((image.shape[:2]), np.uint8)
        cv2.circle(circle_img,(circle_x,circle_y),circle_radius,(255,255,255),-1)
        datos_rgb = cv2.mean(image, mask=circle_img)
        average_value.append(datos_rgb[:nchannels])
    return np.array(average_value)    