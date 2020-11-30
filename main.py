import csv
from varname import nameof
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

with open('./parkinsons.data', newline='') as f:
    reader = csv.reader(f)
    fileData = np.array(list(reader))

attributes = fileData[0]
X = np.delete(fileData[1:], [0, 17], 1).astype(float)
y = fileData[1:, 17].astype(float)

# DANE SYNTETYCZNE
synthetic2ClassX, synthetic2ClassY = make_classification(
    n_samples=5000,
    n_classes=2,
    n_features=10,
    n_redundant=2,
    random_state=1410
)

synthetic3ClassX, synthetic3ClassY = make_classification(
    n_samples=5000,
    n_classes=3,
    n_clusters_per_class=2,
    n_informative=5,
    n_features=10,
    n_redundant=2,
    random_state=1410
)

synthetic2ClassUnbalancedX, synthetic2ClassUnbalancedY = make_classification(
    n_samples=5000,
    n_classes=2,
    weights=[0.3, 0.7],
    n_clusters_per_class=2,
    n_features=10,
    n_redundant=2,
    random_state=1410
)

synthetic3ClassUnbalancedX, synthetic3ClassUnbalancedY = make_classification(
    n_samples=5000,
    n_classes=3,
    weights=[0.1,0.6,0.3],
    n_clusters_per_class=2,
    n_informative=5,
    n_features=10,
    n_redundant=2,
    random_state=1410
)
##############

# print(attributes, )
# print(y, y.shape)
# print(X, X.shape)

pca = PCA(n_components=15)
extractedXPCA = pca.fit_transform(X)

# lda = LinearDiscriminantAnalysis(n_components=15)
# extractedDataLDA = lda.fit_transform(data, classes)

# klasyfikatory
knn = KNeighborsClassifier()
gnb = GaussianNB()
cart = DecisionTreeClassifier(random_state=0)
svc = SVC()
mlp = MLPClassifier()
# + 2 inne sieci


def makeExperiment(expX, expY, expClf):
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1410)
    results = []

    for train_index, test_index in kf.split(expX, expY):
        X_train, X_test = expX[train_index], expX[test_index]
        y_train, y_test = expY[train_index], expY[test_index]
        expClf.fit(X_train, y_train)
        result = expClf.predict(X_test)

        score = accuracy_score(y_test, result)
        results.append(score)

    print(np.mean(results))


clfs = [knn, gnb, cart, svc, mlp]
Xs = [X, extractedXPCA]

for _, clf in enumerate(clfs):
    for _, x in enumerate(Xs):
        makeExperiment(x, y, clf)


# dane syntetyczne
Xs = [synthetic2ClassX, synthetic3ClassX, synthetic2ClassUnbalancedX, synthetic3ClassUnbalancedX]
ys = [synthetic2ClassY, synthetic3ClassY, synthetic2ClassUnbalancedY, synthetic3ClassUnbalancedY,
      synthetic2ClassY, synthetic3ClassY, synthetic2ClassUnbalancedY, synthetic3ClassUnbalancedY]

for _, clf in enumerate(clfs):
    print(clf.__class__)
    for i, x in enumerate(Xs):
        print(i)
        if i >= len(Xs):
            print('EXTRACTED')
            pca = PCA(n_components=3)
            extractedX = pca.fit_transform(x);
            x = extractedX
        makeExperiment(x, ys[i], clf)

