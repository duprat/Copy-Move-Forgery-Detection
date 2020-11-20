# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:10:36 2020

@author: Th0ma
"""
import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np


def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)


def Harris_simple(filepath):
    
    img = cv.imread(filepath,cv.IMREAD_UNCHANGED)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    #displaying
    cv.imshow('Harris Simple',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
        
        
def Harris_Accurate(filepath):
    img = cv.imread(filepath,cv.IMREAD_UNCHANGED)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    
    #And display everything
    cv.imshow("Harris Accurate",img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
        
def Shi_Tomasi(filepath):
    
    
        img = cv.imread(filepath,cv.IMREAD_UNCHANGED)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv.circle(img,(x,y),3,255,-1)
        plt.imshow(img),plt.show()    
        
        
def SIFT(filepath):
    img = cv.imread(filepath,cv.IMREAD_UNCHANGED)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray,None)
    img=cv.drawKeypoints(gray,keyPoints,img)
    cv.imshow("SIFT",img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
        

# inutilisable pour des questions de droits
def SURF(filepath, hessian_threshold):
    img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
    
    surf = cv.xfeatures2d.SURF_create(hessian_threshold)
    
    keyPoints, descriptors = surf.detectAndCompute(img,None)
    
    img2 = cv.drawKeypoints(img,keyPoints,None,(255,0,0),4)
    
    cv.imshow("SURF",img2)
    
    surf.upright = True
    
    keyPoints = surf.detect(img,None)

    img2 = cv.drawKeypoints(img,keyPoints,None,(255,0,0),4)   
    
    cv.imshow("SURF (Orientation discarded)",img2)
    
    surf.extended = True
    
    keyPoints, descriptors = surf.detectAndCompute(img,None)
    
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
        
def main():
    filepath = "../CVIP/Dataset 0/im2_t.bmp"
    
    if isValid(filepath):
        SURF(filepath, 400)
        
        
main()