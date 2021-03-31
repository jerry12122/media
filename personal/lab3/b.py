import numpy as np
import cv2
image = cv2.imread('agar.jpg') #灰階載入
edges = cv2.Canny(image,130,150)
ret,erosion = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)  #二值化
circles = cv2.HoughCircles(erosion,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=50,minRadius = 2,maxRadius=100)
cv2.imshow("image", image)
    
for i in range(0,int(circles.size/3)):
    cv2.circle(image,(circles[0][i][0], circles[0][i][1]), int(circles[0][i][2]), (255, 0, 0), 3)

cv2.imshow("line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
