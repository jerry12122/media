import numpy as np
import cv2

image = cv2.imread('floor.jpg',0) #灰階載入
image = cv2.pyrDown(image)  #縮小
image = cv2.pyrDown(image)  #縮小
image = cv2.medianBlur(image,3)
edges = cv2.Canny(image,130,150)
ret,erosion = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)  #二值化
kernel = np.ones((3,3), np.uint8)
dilate = cv2.dilate(erosion, kernel, iterations = 1) #膨脹
erode = cv2.erode(dilate, kernel, iterations = 1) #膨脹
ret,dilate = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY_INV)#反向二值化 
image = cv2.imread('floor.jpg')#原圖載入
image = cv2.pyrDown(image)#縮小
image = cv2.pyrDown(image)#縮小
lines = cv2.HoughLinesP(erode, 0.1, np.pi/180, threshold = 10, minLineLength = 0, maxLineGap = 30)

for i in lines:
    cv2.line(image, (i[0][0],i[0][1]), (i[0][2],i[0][3]), (0,0,255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
