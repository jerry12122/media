import cv2
import numpy as np

image =  cv2.imread("coin2.jpg")
image = cv2.resize(image, (1000,562), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
total = 0

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1, 60, param1=130, param2=18, minRadius=10, maxRadius=50)
print(circles)
for i in circles[0]:
    if((i[0] < 344 or i[0] > 662) or (i[1] < 285)):
        continue

    if(i[2] < 29.5):
        cv2.rectangle(image, (i[0]-i[2],i[1]-i[2]), (i[0]+i[2],i[1]+i[2]),(0,0,255), 2, 1)
        total += 1
    elif(i[2] <= 35):
        cv2.rectangle(image, (i[0]-i[2],i[1]-i[2]), (i[0]+i[2],i[1]+i[2]),(0,97,255), 2, 1)
        total += 5
    elif(i[2] <= 37):
        cv2.rectangle(image, (i[0]-i[2],i[1]-i[2]), (i[0]+i[2],i[1]+i[2]),(0, 255, 255), 2, 1)
        total += 10
    else:
        cv2.rectangle(image, (i[0]-i[2],i[1]-i[2]), (i[0]+i[2],i[1]+i[2]),(0,255,0), 2, 1)
        total += 50

gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((11,11)))
gray = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
num_label, label, stat, centroid = cv2.connectedComponentsWithStats(gray,connectivity=4)

for i in stat:
    if(i[4] < 1000 or i[4] > 2000):
        continue

    if(i[4] < 1290):
        cv2.rectangle(image, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (255,0,0), 2)
        total += 100
    elif(i[4] < 1300):
        cv2.rectangle(image, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (255,0,255), 2)
        total += 500
    else:
        cv2.rectangle(image, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (255,255,255), 2)
        total += 1000

cv2.putText(image, str(total), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)    
cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()