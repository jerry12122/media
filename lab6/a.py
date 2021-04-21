import imutils
import os
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imutils import paths

train_path = "train"
training_names = os.listdir(train_path)
#print(training_names)

image_paths = []
image_classes = []
class_id = 0

for training_name in training_names:
    dir = os.path.join(train_path,training_name)
    class_path = list(paths.list_images(dir))
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1 

sift = cv2.xfeatures2d.SIFT_create()

des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    im = cv2.resize(im,(300,300))
    kpts = sift.detect(im)
    kpts , des = sift.compute(im,kpts)
    des_list.append((image_path,des))
    print("image file path :",image_path)

descriptors = des_list[0][1]

for image_path,descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors , descriptor))

k = 20
voc,variance = kmeans(descriptors,k,1)

im_features = np.zeros((len(des_list),k),"float32")
for i in range(len(des_list)):
    words,distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

X=im_features
y = np.array(image_classes)
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

clf = LinearSVC()
clf.fit(X_train, y_train)

print("perdict")
print(clf.predict(X_train))
print(clf.predict(X_test))

print("accuracy")
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


'''
def makelist(image, ans):
    targets.append(ans)
    image = transform.resize(image, (620, 470))
    images.append(image)
    gray = cv.cvtColor(image)
    data, hogs = hog(image, orientations = 8, pixels_per_cell = (10, 10), cells_per_block = (5, 5), visualize = True)
    hog_image.append(hogs)
    datas.append(data) 

for i in range(10):
    makelist(io.imread("000%02d.jpg" % (i + 1)), 0)
for i in range(10):
    makelist(people.images[i, :, :], 1)


X_train, X_test, y_train, y_test = train_test_split(datas, targets, test_size = 0.5, random_state = 0)

'''

