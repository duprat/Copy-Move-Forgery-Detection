import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from sklearn import linear_model
from sklearn.cluster import DBSCAN
import copy
import tkinter as tk
from PIL import Image, ImageTk
from functools import partial


def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)

def drawTriangles(img, keyPoints, triangles):
    imgAffichage = copy.deepcopy(img)
    
    for j in keyPoints[triangles.simplices]:
        cpt = 0
        pt1 = (j[cpt][0], j[cpt][1])
        pt2 = (j[cpt+1][0], j[cpt+1][1])
        pt3 = (j[cpt+2][0], j[cpt+2][1])

        v1 = [j[cpt][0], j[cpt][1]]
        v2 = [j[cpt+1][0], j[cpt+1][1]]
        v3 = [j[cpt+2][0], j[cpt+2][1]]
        triangle = np.array([v1, v2, v3])
        cv.polylines(imgAffichage, [triangle], isClosed=True,
                     color=(0, 0, 255), thickness=2)
        cv.circle(imgAffichage, pt1, 2, (0, 0, 255), 3)
        cv.circle(imgAffichage, pt2, 2, (0, 0, 255), 3)
        cv.circle(imgAffichage, pt3, 2, (0, 0, 255), 3)
           
    return imgAffichage
        
def DelaunayTriangulation(image, keyPoints, root):
    
    img = copy.deepcopy(image)
    tri = Delaunay(keyPoints)
     
    imTri = drawTriangles(img, keyPoints, tri)
    
    imTri = cv.cvtColor(imTri, cv.COLOR_BGR2RGB)
    
           
    tk_img =  ImageTk.PhotoImage(image=Image.fromarray(imTri))
    canvas = tk.Canvas(root,width=imTri.shape[1] + 100,height=imTri.shape[0] )
    
    canvas.pack()
    
    button1 = tk.Button (root, text='Needs Ransac',command=partial(ransac,img,keyPoints,tri,root),bg='brown',fg='white')
    canvas.create_window(imTri.shape[1] + 60, 20, window=button1)
    canvas.create_image(10,10, anchor="nw", image=tk_img)
    
    root.mainloop()
         
def ransac(img,keyPoints,tri, root):
    
    
    vm = []
    vmRGB = []
    vmDistance = []
    matchX = []
    matchY = []
    threshold = 3
    thresholdD = 3
    tailleMin = 0
    
    for i in keyPoints[tri.simplices]:
        cpt = 0

        v1 = [i[cpt][0], i[cpt][1]]
        v2 = [i[cpt+1][0], i[cpt+1][1]]
        v3 = [i[cpt+2][0], i[cpt+2][1]]

        x = round(float(v1[0])/3.0+float(v2[0])/3.0+float(v3[0])/3.0)
        y = round(float(v1[1])/3.0+float(v2[1])/3.0+float(v3[1])/3.0)
        xM = abs(x - v1[0])
        yM = abs(y - v1[1])
        if(xM >= tailleMin) and (yM >= tailleMin):
            vm.append([x, y])
            
            vmDistance.append([xM, yM])
            r = round(
                (img[i[cpt][1], i[cpt][0]][0])/3.0 +
                (img[i[cpt+1][1], i[cpt+1][0]][0])/3.0 +
                (img[i[cpt+2][1], i[cpt+2][0]][0])/3.0)


            g = round(
                (img[i[cpt][1], i[cpt][0]][1])/3.0 +
                (img[i[cpt+1][1], i[cpt+1][0]][1])/3.0 +
                (img[i[cpt+2][1], i[cpt+2][0]][1])/3.0)

            b = round((
                (img[i[cpt][1], i[cpt][0]][2])/3.0 +
                (img[i[cpt+1][1], i[cpt+1][0]][2])/3.0 +
                (img[i[cpt+2][1], i[cpt+2][0]][2])/3.0))

            vmRGB.append([r, g, b])

    cpt += 1
  
    for i in range(0, len(vm)-1):
        for j in range(i+1, len(vm)-1):
            x = int(vm[i][1])
            y = int(vm[i][0])
            x2 = int(vm[j][1])
            y2 = int(vm[j][0])
            if(
            abs(vmDistance[i][0] - vmDistance[j][0]) <= thresholdD 
            and (abs(vmDistance[i][1] - vmDistance[j][1]) <= thresholdD)
            and (abs((int(img[x, y][0]) - int(img[x2, y2][0]))) <= threshold)
            and (abs((int(img[x, y][1]) - int(img[x2, y2][1]))) <= threshold)
            and (abs((int(img[x, y][2]) - int(img[x2, y2][2]))) <= threshold)
            ):
                matchX.append([vm[i][0]])
                matchX.append([vm[j][0]])
                matchY.append([vm[i][1]])
                matchY.append([vm[j][1]])

    ransac = linear_model.RANSACRegressor()
    tempArray = np.array(matchX)
    lengthArray = tempArray.size
    nbTest = 1
    dividende = lengthArray/(nbTest+1)
    if dividende == 0:
        dividende = 1
    plt.gca().invert_yaxis()
    for i in range(1, int(lengthArray/dividende)):
        firstSlice = int((i-1)*dividende)
        lastSlice = int(i*dividende)

        Xarray = np.array(matchX[firstSlice:lastSlice])
        Yarray = np.array(matchY[firstSlice:lastSlice])

        ransac.fit(Xarray, Yarray)
        
        inlier_mask = ransac.inlier_mask_

        for j in range(0, Xarray[inlier_mask].size - 1):
            pt = (Xarray[inlier_mask][j][0], Yarray[inlier_mask][j][0])
            pt2 = (Xarray[inlier_mask][j+1][0], Yarray[inlier_mask][j+1][0])

            cv.line(img, pt, pt2, (255, 0, 255), 4)
            cv.circle(img, pt, 4, (255, 255, 0), -1)
            cv.circle(img, pt2, 4, (255, 255, 0), -1)
            
            j += 1
            
            
    cv.imshow("Forgery",img) 
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()    
        
    root.destroy()

def siftDetector(image):
    sift = cv.SIFT_create()
    gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY) 
    key_points,descriptors = sift.detectAndCompute(gray, None)
    return key_points,descriptors
   
def locateForgery(image,key_points,descriptors, root, radius=40,min_sample=2):
    
    forgery = copy.deepcopy(image)
    
    clusters = DBSCAN(eps=radius, min_samples=min_sample).fit(descriptors)
    
    labels = clusters.labels_                       # il y a un label pour chaque point d'interet
    
    size = np.unique(labels).shape[0]-1             # donne le nombre de clusters (sans compter le cluster du bruit)
    
    
    
    if (size==0) and (np.unique(labels)[0]==-1):    # il n'y a que du bruit
        print('No Forgery Found!!')
        return None
    
    if size==0:                                     # il n'y a qu'un cluster
        size=1
        
    cluster_list= [[] for i in range(size)]         # initialise la liste au nombre de clusters
    
    for idx in range(len(key_points)):              # on parcours tous les points d'interet
        
        if labels[idx]!=-1:                         # si le point n'est pas du bruit
        
            # pour chaque cluster, on lui ajoute les coordonnees spatiales des points lui appartenant
            cluster_list[labels[idx]].append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))        
    
    
    
    points_for_delaunay = list()
    
    for points in cluster_list:
       if len(points)>1: 
           for idx1 in range(len(points)):
               points_for_delaunay.append(points[idx1])
    
    if(len(points_for_delaunay) > 4):
        DelaunayTriangulation(image, np.array(points_for_delaunay),root)
    else:
        print("Delaunay Not Needed")
        for points in cluster_list:
            if len(points)>1:                           # s'il y a plus d'un point dans le cluster
                for idx1 in range(1,len(points)):       # on parcourt les points du cluster
                   # on trace une ligne entre le premier point et tous les autres
                   cv.line(forgery,points[0],points[idx1],(255,0,0),5)
    
        cv.imshow("Forgery", forgery)

        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()   
     
def main(filepath):
    root = tk.Tk()

    img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
    
    keyPoints, descriptors = siftDetector(img)
    
    
    locateForgery(img, keyPoints, descriptors, root)



filepath = "../CVIP/Dataset 0/im17_t.bmp"

if isValid(filepath):
    
    main(filepath)

else:
    print("file not valid")