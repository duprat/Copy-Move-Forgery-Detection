
from scipy.spatial import Delaunay
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


points = np.array([[100, 50], [68, 20], [100, 0], [50, 500],[150, 250], [68, 250], [150, 48], [105, 100]])

tri = Delaunay(points)
img = cv.imread('./dataset/Dataset 0/im2_t.bmp')
vm = []
matchX = []
matchY = []
threshold = 1
for i in points[tri.simplices]:
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



lw = 2
plt.scatter(Xarray[inlier_mask], Yarray[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(Xarray[outlier_mask], Yarray[outlier_mask], color='gold', marker='.',
            label='Outliers')

plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')

plt.triplot(points[:,0], points[:,1], tri.simplices)

plt.plot(points[:,0], points[:,1], 'o')
plt.show()

