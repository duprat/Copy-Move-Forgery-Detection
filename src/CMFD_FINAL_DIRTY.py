import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from sklearn import linear_model, datasets
from torchvision import models
import torchvision.transforms as T
from PIL import Image
import torch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import copy


def segment(net,  path):
    img = Image.open(path)
    trf = T.Compose([T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om, path)
    
    return rgb
   


def decode_segmap(image, source, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)

    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    
    # Load the foreground input image
    foreground = cv.imread(source)
    
    # Change the color of foreground image to RGB
    # and resize image to match shape of R-band in RGB output map
    foreground = cv.cvtColor(foreground, cv.COLOR_BGR2RGB)
    foreground = cv.resize(foreground,(r.shape[1],r.shape[0]))
    
    # Create a background array to hold white pixels
    # with the same size as RGB output map
    background = 255 * np.ones_like(rgb).astype(np.uint8)
    
    
    
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv.threshold(np.array(rgb),0,255, cv.THRESH_BINARY)
    
    plt.imshow(background)
    plt.axis('off')
    plt.show()
    
    # Apply a slight blur to the mask to soften edges
    alpha = cv.GaussianBlur(alpha, (7,7),0)
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    
    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)
    
    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)
    
    # Add the masked foreground and background
    outImage = cv.add(foreground, background)
    
    # Return a normalized output image for display
    return outImage/255

def PreProcessing1(img):
    yuv_img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    yuv_img[:, :, 0] = cv.equalizeHist(yuv_img[:, :, 0])
    processed_img = cv.cvtColor(yuv_img, cv.COLOR_YUV2BGR)
    return processed_img


def PreProcessing2(img):
    yuv_img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    yuv_img[:, :, 0] = cv.equalizeHist(yuv_img[:, :, 0])
    equalized = cv.cvtColor(yuv_img, cv.COLOR_YUV2BGR)
    processed_img = cv.fastNlMeansDenoisingColored(
        equalized, None, 10, 10, 7, 21)
    return processed_img


def PreProcessing3(img):
    denoized = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    yuv_img = cv.cvtColor(denoized, cv.COLOR_BGR2YUV)
    yuv_img[:, :, 0] = cv.equalizeHist(yuv_img[:, :, 0])
    processed_img = cv.cvtColor(yuv_img, cv.COLOR_YUV2BGR)

    return processed_img


def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)

"""
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv.circle( img, p, 2, color, cv.cv.CV_FILLED, cv.CV_AA, 0 )


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.CV_AA, 0)
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.CV_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.CV_AA, 0)


def otherDelaunay(img):
    w = img.shape[1]
    h = img.shape[0]
    rectangle = (0,0,w,h)
    subdivision = cv.Subdiv2D(rectangle)
    triangles = subdivision.getTriangleList()
"""

def DelaunayTriangulation_old(img, keyPoints, imgEcrite):
    
    tri = Delaunay(keyPoints)
    

    vm = []
    vmRGB = []
    vmDistance = []
    matchX = []
    matchY = []
    threshold = 2
    thresholdD = 2
    tailleMin = 0

    '''
    for j in keyPoints[tri.simplices]:
        cpt = 0
        pt1 = (j[cpt][0], j[cpt][1])
        pt2 = (j[cpt+1][0], j[cpt+1][1])
        pt3 = (j[cpt+2][0], j[cpt+2][1])

        v1 = [j[cpt][0], j[cpt][1]]
        v2 = [j[cpt+1][0], j[cpt+1][1]]
        v3 = [j[cpt+2][0], j[cpt+2][1]]
        triangle = np.array([v1, v2, v3])
        cv.polylines(imgEcrite, [triangle], isClosed=True,
                     color=(0, 0, 255), thickness=1)
        cv.circle(imgEcrite, pt1, 2, (0, 0, 255), 1)
        cv.circle(imgEcrite, pt2, 2, (0, 0, 255), 1)
        cv.circle(imgEcrite, pt3, 2, (0, 0, 255), 1)
        '''
        
        
    # on montre les triangles obtenus    
  #  cv.imshow("Triangles",imgEcrite)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    
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

    print("HERE")
    lw = 2
    ransac = linear_model.RANSACRegressor()
    tempArray = np.array(matchX)
    lengthArray = tempArray.size
    nbTest = 1
    dividende = lengthArray/(nbTest+1)
    plt.gca().invert_yaxis()
    for i in range(1, int(lengthArray/dividende)):
        firstSlice = int((i-1)*dividende)
        lastSlice = int(i*dividende)

        Xarray = np.array(matchX[firstSlice:lastSlice])
        Yarray = np.array(matchY[firstSlice:lastSlice])

        ransac.fit(Xarray, Yarray)
        
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        '''
        while(j > 100):
            Xarray = Xarray[outlier_mask]
            Yarray = Yarray[outlier_mask]
            ransac.fit(Xarray,Yarray)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            j = Xarray[outlier_mask].size
        

        
        for j in range(0, Xarray[outlier_mask].size - 1):
            pt = (Xarray[outlier_mask][j][0], Yarray[outlier_mask][j][0])
            pt2 = (Xarray[outlier_mask][j+1][0], Yarray[outlier_mask][j+1][0])
            cv.circle(img, pt, 2, (255, 255, 0), -1)
            cv.circle(img, pt2, 2, (255, 255, 0), -1)
            cv.line(img, pt, pt2, (255, 255, 0), 1)
            j += 1
        '''

        for j in range(0, Xarray[inlier_mask].size - 1):
            pt = (Xarray[inlier_mask][j][0], Yarray[inlier_mask][j][0])
            pt2 = (Xarray[inlier_mask][j+1][0], Yarray[inlier_mask][j+1][0])

            cv.circle(img, pt, 2, (255, 255, 0), -1)
            cv.circle(img, pt2, 2, (255, 255, 0), -1)
            cv.line(img, pt, pt2, (255, 0, 255), 1)
            j += 1


def DelaunayTriangulation(img, keyPoints, imgEcrite, imgMask):
    #cv.imshow("DbScan", img)
    tri = Delaunay(keyPoints)      
    
    vm = []
    vmRGB = []
    vmDistance = []
    matchX = []
    matchY = []
    threshold = 4
    thresholdD = 4
    tailleMin = 0

    if (cv.countNonZero(cv.cvtColor(imgMask, cv.COLOR_BGR2GRAY)) == 0):
            isImgBlack = True
            print("La détection de l'IA n'a pas fonctionnée, méthode 2 utilisée")
    else:
            isImgBlack = False

    for i in keyPoints[tri.simplices]:
        cpt = 0

        v1 = [i[cpt][0], i[cpt][1]]
        v2 = [i[cpt+1][0], i[cpt+1][1]]
        v3 = [i[cpt+2][0], i[cpt+2][1]]

        x = round(float(v1[0])/3.0+float(v2[0])/3.0+float(v3[0])/3.0)
        y = round(float(v1[1])/3.0+float(v2[1])/3.0+float(v3[1])/3.0)
        xM = abs(x - v1[0])
        yM = abs(y - v1[1])
       

        if(xM >= tailleMin) and (yM >= tailleMin) and ( isImgBlack or (int(imgMask[y, x][0]) > 0) or (int(imgMask[y, x][1]) > 0) or (int(imgMask[y, x][2]) > 0)) :
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

    #

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


    lw = 2
    ransac = linear_model.RANSACRegressor()
    tempArray = np.array(matchX)
    lengthArray = tempArray.size
    nbTest = 1
    dividende = lengthArray/(nbTest+1)
    plt.gca().invert_yaxis()
    if (int(dividende) == 0):
    	print("Erreur, mauvaise détection de foreground")
    	return -1
    for i in range(1, int(lengthArray/dividende)):
        firstSlice = int((i-1)*dividende)
        lastSlice = int(i*dividende)

        Xarray = np.array(matchX[firstSlice:lastSlice])
        Yarray = np.array(matchY[firstSlice:lastSlice])

        ransac.fit(Xarray, Yarray)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        '''
        if(isImgBlack):
            while(j > 100):
                Xarray = Xarray[outlier_mask]
                Yarray = Yarray[outlier_mask]
                ransac.fit(Xarray, Yarray)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
                j = Xarray[outlier_mask].size
            
            for j in range(0, Xarray[outlier_mask].size - 1):
                pt = (Xarray[outlier_mask][j][0], Yarray[outlier_mask][j][0])
                pt2 = (Xarray[outlier_mask][j+1][0], Yarray[outlier_mask][j+1][0])
                cv.circle(img, pt, 2, (255, 255, 0), -1)
                cv.circle(img, pt2, 2, (255, 255, 0), -1)
                cv.line(img, pt, pt2, (255, 255, 0), 1)
                j += 1
        else:
        '''
        for j in range(0, Xarray[inlier_mask].size - 1):
            pt = (Xarray[inlier_mask][j][0], Yarray[inlier_mask][j][0])
            pt2 = (Xarray[inlier_mask][j+1][0], Yarray[inlier_mask][j+1][0])

            cv.circle(img, pt, 2, (255, 255, 0), -1)
            cv.circle(img, pt2, 2, (255, 255, 0), -1)
            cv.line(img, pt, pt2, (255, 0, 255), 1)
            j += 1

def siftDetector(image):
 sift = cv.SIFT_create()
 gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY) 
 key_points,descriptors = sift.detectAndCompute(gray, None)
 return key_points,descriptors
        
def locateForgery(image,key_points,descriptors,radius=40,min_sample=2):
    
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
    """        
    for points in cluster_list:
       
       if len(points)>1: 
           for idx1 in range(len(points)):
              cv.circle(forgery,points[idx1], 5, (0,255,255),-1)
    """           
    
    points_for_delaunay = list()
    
    for points in cluster_list:
       if len(points)>1: 
           for idx1 in range(len(points)):
               points_for_delaunay.append(points[idx1])
    
    imgEcrite = copy.deepcopy(image)                # avec une copy simple il y a risque qu'image ecrite et image
                                                    # continuent a avoir les mêmes valeurs 
    if(len(points_for_delaunay) > 4):
        DelaunayTriangulation_old(image, np.array(points_for_delaunay), imgEcrite)
    
     #cv.imshow("Delaunay Triangulation", image)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
            
            
    for points in cluster_list:
       
       if len(points)>1:                           # s'il y a plus d'un point dans le cluster

           for idx1 in range(1,len(points)):       # on parcourt les points du cluster
           
               # on trace une ligne entre le premier point et tous les autres
               cv.line(forgery,points[0],points[idx1],(255,0,0),4)
    
    
    """
    clusters_centroids = [(0,0) for i in range(size)]
    
    accu = 0
    for points in cluster_list:
       
       if len(points)>1:                           # s'il y a plus d'un point dans le cluster
           cx = points[0][0]
           cy = points[0][1]
           for idx1 in range(1,len(points)):       # on parcourt les points du cluster
           
               # on trace une ligne entre le premier point et tous les autres
               cv.line(forgery,points[0],points[idx1],(255,0,0),4)
               
               
               cx = cx + points[idx1][0]
               cy = cy + points[idx1][1]
               
       clusters_centroids[accu] = (int(cx / len(points)),int(cy / len(points)))
       
       accu += 1
                
    centroids_clusters = DBSCAN(eps=radius, min_samples=min_sample).fit(clusters_centroids)
    
    centroids_labels = centroids_clusters.labels_
    
    print(centroids_labels)
    
    centroids_size = np.unique(centroids_labels).shape[0]-1 
    
    
    
    for i in range(centroids_size):
        if(centroids_labels[i] != -1):
            cv.circle(forgery,clusters_centroids[i], 5, (0,255,255),-1)
    """
    if(len(points_for_delaunay) < 5):
        return forgery
    else:
        return image
'''
def main(filepath):
      #fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
      dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
      
      img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
      gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
      imgEcrite = copy.deepcopy(img)
      imgAndKeyFeatures = copy.deepcopy(img)

      imgMask = segment(dlab,filepath)
      
      
      #img = PreProcessing3(img)
      

      sift = cv.SIFT_create()

      keyPoints = sift.detect(gray)
      descriptors = sift.compute(gray, keyPoints)
     
      cv.drawKeypoints(img, keyPoints, imgAndKeyFeatures,
                       flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
      
      #cv.imshow("Key Points", imgAndKeyFeatures)

      if cv.waitKey(0) & 0xff == 27:
          cv.destroyAllWindows()
      
      points = list()
      for keypoint in keyPoints:
          x = int(keypoint.pt[0])
          y = int(keypoint.pt[1])

          points.append((x, y))

      points = np.array(points)
      
      DelaunayTriangulation(img, points, imgEcrite, imgMask)

      cv.imshow("Delaunay Triangulation after Ransac", img)

      if cv.waitKey(0) & 0xff == 27:
          cv.destroyAllWindows()

'''
def mainDBSCAN(filepath):
    img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
    imgEcrite = copy.deepcopy(img)
    
    keyPoints, descriptors = siftDetector(img)
    
    forgery = locateForgery(img, keyPoints, descriptors, 40,2)

    cv.imshow("Forgery", forgery)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


filepath = "./dataset/Dataset 0/im2_t.bmp"
if isValid(filepath):

    #main(filepath)
    mainDBSCAN(filepath)

else:
    print("file not valid")
