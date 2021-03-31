import numpy as np
import cv2

image = cv2.imread('coin.jpg',0) #灰階載入
image = cv2.pyrDown(image)  #縮小
image = cv2.pyrDown(image)  #縮小
ret,erosion = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  #二值化
kernel = np.ones((3,3), np.uint8)
dilate = cv2.dilate(erosion, kernel, iterations = 3) #膨脹
ret,dilate = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY_INV)#反向二值化 
circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius = 10,maxRadius=100)  #圓形偵測
image = cv2.imread('coin.jpg')#原圖載入
image = cv2.pyrDown(image)#縮小
image = cv2.pyrDown(image)#縮小
print(circles)
for i in range(0,int(circles.size/3)):
    
    x1 = int(circles[0][i][0])+int(circles[0][i][2])
    y1 = int(circles[0][i][1])+int(circles[0][i][2])
    x2 = int(circles[0][i][0])-int(circles[0][i][2])
    y2 = int(circles[0][i][1])-int(circles[0][i][2])
    if(circles[0][i][2]>62.5):
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)#畫方形
    elif(circles[0][i][2]>55):
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)#畫方形
    elif(circles[0][i][2]>48):
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,97,255),2)#畫方形
    else:
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)#畫方形
    cv2.imshow('Input2', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
