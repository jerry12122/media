
import cv2
import numpy as np

apple =  cv2.imread("apple.jpg")
orange =  cv2.imread("orange.jpg")

for i in range(3):
    apple = cv2.pyrDown(apple)
    orange = cv2.pyrDown(orange)
    apple = cv2.resize(apple, (512,512), interpolation = cv2.INTER_CUBIC)
    orange = cv2.resize(orange, (512,512), interpolation = cv2.INTER_CUBIC)
apple = apple[0:512, 0:256]
orange = orange[0:512, 256:512]
combine = np.hstack((apple, orange))

for i in range(10):
	combine = cv2.pyrUp(combine)
	combine = cv2.pyrDown(combine)

cv2.imshow("combine", combine)
cv2.waitKey(0)
cv2.destroyAllWindows()    
