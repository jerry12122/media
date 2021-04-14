from skimage.feature import hog
from skimage import data , exposure
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from skimage import transform
from skimage import io

people = datasets.fetch_lfw_people()

images = []
hog_image = []
datas = []
targets = []

def makelist(image, ans):
    targets.append(ans)
    image = transform.resize(image, (620, 470))
    images.append(image)
    data, hogs = hog(image, orientations = 8, pixels_per_cell = (10, 10), cells_per_block = (5, 5), visualize = True)
    hog_image.append(hogs)
    datas.append(data) 

for i in range(10):
    makelist(io.imread("000%02d.jpg" % (i + 1)), 0)
for i in range(10):
    makelist(people.images[i, :, :], 1)


X_train, X_test, y_train, y_test = train_test_split(datas, targets, test_size = 0.5, random_state = 0)

clf = svm.SVC(kernel = 'poly', C = 1, gamma = 'auto')
clf.fit(X_train, y_train)

print("perdict")
print(clf.predict(X_train))
print(clf.predict(X_test))

print("accuracy")
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
