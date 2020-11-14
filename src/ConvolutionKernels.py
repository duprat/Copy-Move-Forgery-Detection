# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:54:34 2020

@author: Th0ma
"""
import math
import numpy as np
from skimage.exposure import rescale_intensity


######contours only on greyScale images
faible3x3 = {"name": "faible3x3", "scalar": 1, "noyau": np.array([[1,0,-1],
                                                        [0,0,0],
                                                        [-1,0,1]])}

moyen3x3 = {"name": "moyen3x3", "scalar": 1, "noyau": np.array([[0,1,0],
                                                       [1,-4,1],
                                                       [0,1,0]])}

fort3x3 = {"name": "fort3x3", "scalar": 1, "noyau":  np.array([[-1,-1,-1],
                                                      [-1,8,-1],
                                                      [-1,-1,-1]])}

###### netteté only on greyscale images
    
nette3x3 = {"name": "nette3x3", "scalar": 1, "noyau":  np.array([[0,-1,0],
                                                       [-1,5,-1],
                                                       [0,-1,0]])}
    
####flous 
box3x3 = {"name": "box3x3", "scalar": 1 / 9,  "noyau":  np.array([[1,1,1],
                                                      [1,1,1],
                                                      [1,1,1]])}

gauss3x3 = {"name": "gauss3x3", "scalar": 1 / 16, "noyau":  np.array([[1,2,1],
                                                            [2,4,2],
                                                            [1,2,1]])}

gauss5x5 = {"name": "gauss5x5", "scalar": 1 / 256, "noyau":  np.array([[1,4,6,4,1],
                                                             [4,16,24,16,4],
                                                             [6,24,36,24,6],
                                                             [4,16,24,16,4],
                                                             [1,4,6,4,1]])}


def isGrey(img):
    return img.ndim == 2

def isRGB(img):
    return img.ndim == 3 and img.shape[2] == 3

def isRGBA(img):
    return img.ndim == 3 and img.shape[2] == 4


def verif(noyau, img):
    if isRGB(img):
        if noyau["name"] == "faible3x3":
            return False
        if noyau["name"] == "moyen3x3":
            return False
        if noyau["name"] == "fort3x3":
            return False
        if noyau["name"] == "nette3x3":
            return False
    return True
            
            
 
def apply(noyau,img):
    
    assert(verif(noyau, img))
        
    
    print("applying {}".format(noyau["name"]))
    
    out = np.zeros(img.shape,img.dtype)
    
    if isRGB(img):
        
        size_y = noyau["noyau"].shape[0]
        size_x = noyau["noyau"].shape[1]
        
        padding_x = math.floor(size_x / 2)
        padding_y = math.floor(size_y / 2)
        
        for y in range(padding_y,img.shape[0] - padding_y):
            for x in range(padding_x,img.shape[1] - padding_x):
                
                offset_x = x - padding_x
                offset_y = y - padding_y
                
                accu_r = 0
                accu_g = 0
                accu_b = 0
                
                scalar = noyau["scalar"]
                
                kernel = noyau["noyau"]
                
                for k_y in range(0,size_y):
                    for k_x in range(0,size_x):
                       accu_r += scalar * (kernel[k_y, k_x] * img[k_y + offset_y, k_x + offset_x,0])
                       accu_g += scalar * (kernel[k_y, k_x] * img[k_y + offset_y, k_x + offset_x,1])
                       accu_b += scalar * (kernel[k_y, k_x] * img[k_y + offset_y, k_x + offset_x,2])
                       
                out[y,x, 0] = accu_r
                out[y,x, 1] = accu_g
                out[y,x, 2] = accu_b
                
        out = rescale_intensity(out, in_range=(0, 255))
        
    elif isGrey(img):
        size_y = noyau["noyau"].shape[0]
        size_x = noyau["noyau"].shape[1]
        
        padding_x = math.floor(size_x / 2)
        padding_y = math.floor(size_y / 2)
        
        for y in range(padding_y,img.shape[0] - padding_y):
            for x in range(padding_x,img.shape[1] - padding_x):
                
                offset_x = x - padding_x
                offset_y = y - padding_y
                
                accu = 0
                
                scalar = noyau["scalar"]
                
                kernel = noyau["noyau"]
                
                for k_y in range(0,size_y):
                    for k_x in range(0,size_x):
                       accu += scalar * (kernel[k_y, k_x] * img[k_y + offset_y, k_x + offset_x])
                       
                out[y,x] = accu
                
        out = rescale_intensity(out, in_range=(0, 255))
        
    return out
