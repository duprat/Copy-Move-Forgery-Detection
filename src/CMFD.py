import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from sklearn import linear_model, datasets



def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)

def DelaunayTriangulation(img,keyPoints):
    tri = Delaunay(keyPoints)
    
    vm = []
    matchX = []
    matchY = []
    threshold = 1
    for i in keyPoints[tri.simplices]:
        cpt = 0
        
        v1 = [i[cpt][0] , i[cpt][1]]
        cpt+=1
        v2 = [i[cpt][0] , i[cpt][1]]
        cpt+=1
        v3 = [i[cpt][0] , i[cpt][1]]
        
        x = round(v1[0]/3+v2[0]/3+v3[0]/3)
        y = round(v1[1]/3+v2[1]/3+v3[1]/3)
        vm.append([x,y])
    
    for i in range(0, len(vm)-1):
        for j in range(i, len(vm)-1):
            x = int(vm[i][0])
            y = int(vm[i][1])
            x2 = int(vm[j][0])
            y2 = int(vm[j][1])
            if(i != j and (img[x, y][0] - img[x2, y2][0] <= threshold) and (img[x, y][1] - img[x2, y2][1] <= threshold) and (img[x, y][2] - img[x2, y2][2] <= threshold)):
                matchX.append([vm[i][0]])
                matchX.append([vm[j][0]])
                matchY.append([vm[i][1]])
                matchY.append([vm[j][1]])
    
    print(matchX)
    print(matchY)
    Xarray = np.asarray(matchX, dtype=np.float32)
    Yarray = np.asarray(matchY, dtype=np.float32)
    
    ransac = linear_model.RANSACRegressor()
    ransac.fit(Xarray, Yarray)
    print(ransac.score(Xarray, Yarray))
    
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    line_X = np.arange(Xarray.min(), Xarray.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

  

def main(filepath): 
    
    
    if isValid(filepath):
        
        img = cv.imread(filepath,cv.IMREAD_UNCHANGED)
        cv.imshow("Image de Base", img)
        
        
        grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        imgAndKeyFeatures = cv.imread(filepath,cv.IMREAD_UNCHANGED)
        
        sift = cv.SIFT_create()
        
        keyPoints = sift.detect(img)
        keyPoints, descriptors = sift.compute(img, keyPoints)
        
        
        
        cv.drawKeypoints(img,keyPoints,imgAndKeyFeatures,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        
        points = list()
        for keypoint in keyPoints:
            points.append((int(keypoint.pt[0]),int(keypoint.pt[1])))
            
         
        points = np.array(points)
        
        print(points)
        
        DelaunayTriangulation(img,points)
        
        
        cv.imshow("Feature Detection", imgAndKeyFeatures)
        
        
        
        #cv.imshow("Delaunay Triangulation", grey )
        
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
    else:
        print("\nsomething seems wrong with your file path...\n")



filepath = "../CVIP/Dataset 0/im2_t.bmp"
#filepath = '../blocs.jpg'

if isValid(filepath):
    
    main(filepath)
    
else:
    print("file not valid")