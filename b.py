import cv2
import numpy as np

img = np.zeros((400,400,3),np.uint8)
img.fill(20)

cv2.line(img,(0,0),(255,255),(0,0,255),5)
cv2.rectangle(img,(20,60),(120,160),(0,0,255),2)
cv2.circle(img,(200,200),30,(0,0,255),3)

cv2.imshow("Draw test",img)

cv2.waitKey(0)
cv2.destroyAllWindows()