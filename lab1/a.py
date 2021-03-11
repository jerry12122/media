import cv2
import numpy as np

img = cv2.imread("a.jpg")
cv2.imshow("cat_ori",img)

img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("cat_GRAY",img2)

img3 = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
cv2.imshow("cat_YCRCB",img3)

img4 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("cat_HSV",img4)

img_r = img[:,:,2]
img_g = img[:,:,1]
img_b = img[:,:,0]
cv2.imshow("cat_r",img_r)
cv2.imshow("cat_g",img_g)
cv2.imshow("cat_b",img_b)

cv2.waitKey(0)
cv2.destroyAllWindows()