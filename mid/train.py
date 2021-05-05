import imutils
import os
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imutils import paths
import joblib
import random
#載入訓練資料集
train_path = "train"
training_names = os.listdir(train_path)
image_paths = []
image_classes = []
class_id = 0

for training_name in training_names:
    dir = os.path.join(train_path,training_name)
    class_path = list(paths.list_images(dir))
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1 
#SIFT特徵提取器
sift = cv2.xfeatures2d.SIFT_create()
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    im = cv2.resize(im,(300,300))
    kpts = sift.detect(im)
    kpts , des = sift.compute(im,kpts)
    des_list.append((image_path,des))

descriptors = des_list[0][1]

for image_path,descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors , descriptor))
#進行k-means分群
k = 30
voc,variance = kmeans(descriptors,k,1)
im_features = np.zeros((len(des_list),k),"float32")
for i in range(len(des_list)):
    words,distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
#進行SVM訓練
X=im_features
y = np.array(image_classes)
clf = LinearSVC()
clf.fit(X, y)

#載入測試資料
train_path = "train"
test_names = os.listdir(train_path)
#隨機產生電腦的手勢
dir = os.path.join(train_path,str(np.random.choice(test_names,1,replace = True)[0]))
class_path = list(paths.list_images(dir))
computer = str(np.random.choice(class_path,1,replace = True)[0])
#輸入使用者的手勢
player = input()
dir = os.path.join(train_path,player)
class_path = list(paths.list_images(dir))
player = str(np.random.choice(class_path,1,replace = True)[0])
#圖像處理函數
def img(name,img):
    des_list2 = []
    im = cv2.imread(img)
    im = cv2.resize(im,(300,300))
    cv2.imshow(name, im)
    kpts = sift.detect(im)
    kpts , des = sift.compute(im,kpts)
    des_list2.append((img,des))

    descriptors = des_list2[0][1]

    for image_path,descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors , descriptor))

    im_features2 = np.zeros((len(des_list),k),"float32")    
    for i in range(len(des_list)):
        words,distance = vq(des_list[i][1],voc)
        for w in words:
            im_features2[i][w] += 1
    pre = clf.predict(im_features2)[0]
    l = ['paper','sciss' , 'rock']
    return [img[6:11], im]
#使用者及電腦進行圖像處理
player_res = img("player",player)
computer_res = img("computer",computer)
#產生比較結果
result = "res/"
if player_res[0] == computer_res[0]:
    result += "pin.png"
elif (player_res[0]=="sciss" and computer_res[0]=="paper") or (player_res[0]=="paper" and computer_res[0]=="stone") or (player_res[0]=="stone" and computer_res[0]=="sciss"):
    result += "win.png"
else:
    result += "lose.png"
#秀圖
combine = np.hstack((player_res[1], cv2.imread(result), computer_res[1]))
cv2.imshow("result", combine)

cv2.waitKey(0)
cv2.destroyAllWindows()