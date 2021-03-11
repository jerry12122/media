import cv2
import numpy as np

image = cv2.imread('man.jpg', 0)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(image, kernel, iterations = 3)
'''kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(erosion, kernel, iterations = 5)
ret,erosion = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY)
'''
result = cv2.subtract(image, erosion)
ret,result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Input', image)
cv2.imshow('Result', result)
cv2.waitKey(0)