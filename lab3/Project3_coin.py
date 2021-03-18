import numpy as np

import cv2


image = cv2.imread('coin.jpg', 0)
image = cv2.pyrDown(image)
image = cv2.pyrDown(image)


ret,erosion = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)   #二值化

blur = cv2.GaussianBlur(erosion, (3, 3), 0)

cani = cv2.Canny(blur, 90, 150)



"""
lines = cv2.HoughLines(cani,1,np.pi/180,200)
"""



"""
kernel = np.ones((3,3), np.uint8)                                   #設定大小
erosion = cv2.erode(image, kernel, iterations = 4)
"""


"""
kernel = np.ones((3,3), np.uint8)
erosion = cv2.dilate(erosion, kernel, iterations = 3)
"""
"""
kernel = np.ones((3,3), np.uint8)                                   #設定大小
erosion = cv2.erode(erosion, kernel, iterations = 4)
"""






cv2.imshow('Input', image)
cv2.imshow('Output', cani)

cv2.waitKey(0)
cv2.destroyAllWindows()



