import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

with open('./parkinsons.data', newline='') as f:
    reader = csv.reader(f)
    fileData = np.array(list(reader))

attributes = fileData[0]
data = np.delete(fileData[1:],[0,17],1)
classes = fileData[1:,17]

print(attributes)
print(classes)
print(data, len(data[0]))

pca = PCA(n_components=15)
extractedData = pca.fit_transform(data)

print(extractedData, len(extractedData[0]))


#klasyfikatory
knn = KNeighborsClassifier()
gnb = GaussianNB()
cart = DecisionTreeClassifier(random_state=0)
svc = SVC()
mlp = MLPClassifier()
# + 2 inne sieci







