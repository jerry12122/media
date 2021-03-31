import numpy as np
import cv2
image = cv2.imread('unnamed.jpg') #灰階載入
edges = cv2.Canny(image,130,150)
ret,erosion = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)  #二值化
lines = cv2.HoughLinesP(erosion, 0.1, np.pi/180, threshold = 10, minLineLength = 0, maxLineGap = 30)
cv2.imshow("image", erosion)

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


cv2.imshow("line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
