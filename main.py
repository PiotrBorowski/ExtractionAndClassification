import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_ind
from tabulate import tabulate

import warnings
from datetime import datetime

# Po ludzku na ekranie
np.set_printoptions(suppress=True, precision=3)

warnings.simplefilter(action='ignore', category=FutureWarning)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



def print_to_file(text):
    with open('result.txt', 'a') as f:
        print(text)
        sys.stdout = f
        print(text)
        sys.stdout = original_stdout


original_stdout = sys.stdout
print_to_file("\n\n\n---------------------------------- NEW TEST - "
              + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
              + " ----------------------------------\n\n\n")

with open('./parkinsons.data', newline='') as f:
    reader = csv.reader(f)
    fileData = np.array(list(reader))

attributes = fileData[0]
X = np.delete(fileData[1:], [0, 17], 1).astype(float)
y = fileData[1:, 17].astype(float)

# DANE SYNTETYCZNE
synthetic2ClassX, synthetic2ClassY = make_classification(
    n_samples=1000,
    n_classes=2,
    n_features=100,
    random_state=1410,
    n_informative=50
)

synthetic3ClassX, synthetic3ClassY = make_classification(
    n_samples=1000,
    n_classes=3,
    n_clusters_per_class=1,
    n_features=100,
    random_state=1410,
    n_informative=50
)

synthetic2ClassUnbalancedX, synthetic2ClassUnbalancedY = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.3, 0.7],
    n_clusters_per_class=2,
    n_features=100,
    n_redundant=2,
    random_state=1410,
    n_informative=50
)

synthetic3ClassUnbalancedX, synthetic3ClassUnbalancedY = make_classification(
    n_samples=1000,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.1, 0.6, 0.3],
    n_features=100,
    random_state=1410,
    n_informative=50
)


def createData(features):
    x, y = make_classification(
        n_samples=1000,
        n_classes=2,
        n_features=features,
        random_state=1410,
        n_informative=round((features / 2))
    )
    return x, y


##############

# print(attributes, )
# print(y, y.shape)
# print(X, X.shape)


# lda = LinearDiscriminantAnalysis(n_components=15)
# extractedDataLDA = lda.fit_transform(data, classes)

# klasyfikatory
knn = KNeighborsClassifier()
gnb = GaussianNB()
cart = DecisionTreeClassifier(random_state=0)
svc = SVC()
mlp = MLPClassifier()
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 100))
mlp3 = MLPClassifier(hidden_layer_sizes=(100, 100, 100))
mlp4 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100))

# + 2 inne sieci
# clfs = [knn, gnb, cart, svc, mlp, mlp2, mlp3, mlp4 ]
clfs = [knn, gnb, cart]

clfsResults = []
clfsResultsBalanced = []


def makeExperiment(expX, expY, expClf):
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1410)
    results = []
    balancedResults = []

    for train_index, test_index in kf.split(expX, expY):
        X_train, X_test = expX[train_index], expX[test_index]
        y_train, y_test = expY[train_index], expY[test_index]
        expClf.fit(X_train, y_train)
        result = expClf.predict(X_test)

        score = accuracy_score(y_test, result)
        balanced_score = balanced_accuracy_score(y_test, result)
        results.append(score)
        balancedResults.append(balanced_score)

    # print_to_file(results)
    clfsResults.append(results)
    # print_to_file(balancedResults)
    clfsResultsBalanced.append(balancedResults)

    return results, balancedResults


# MMO
pca = PCA(n_components=15)
pca.fit(X[:round(len(X) * 0.8)])
extractedXPCA = pca.transform(X)

Xs = [X, extractedXPCA]


#
# for i, x in enumerate(Xs):
#     print_to_file(i)
#     for _, clf in enumerate(clfs):
#         print_to_file(clf)
#         makeExperiment(x, y, clf)

def test(x):
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))
    print(p_value.shape, len(x))
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            print((i, j))
            t_statistic[i, j], p_value[i, j] = ttest_ind(x[i], x[j])

    return t_statistic, p_value


def printTest(t, p):
    headers = ['knn', 'gnb', 'cart', 'svc', 'mlp', 'mlp2', 'mlp3', 'mlp4']
    names_column = [['knn'], ['gnb'], ['cart'], ['svc'], ['mlp'], ['mlp2'], ['mlp3'], ['mlp4']]
    t_statistic_table = np.concatenate((names_column, t), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


# print('bez ekstr')
# printTest(*test(clfsResults[:8]))
# print('z ekstr')
# printTest(*test(clfsResults[8:]))
# print('----------BALANCE_ACCURACY-----------')
# print('bez ekstr')
# printTest(*test(clfsResultsBalanced[:8]))
# print('z ekstr')
# printTest(*test(clfsResultsBalanced[8:]))

# dane syntetyczne PON
Xs = [synthetic2ClassX, synthetic3ClassX, synthetic2ClassUnbalancedX, synthetic3ClassUnbalancedX]
ys = [synthetic2ClassY, synthetic3ClassY, synthetic2ClassUnbalancedY, synthetic3ClassUnbalancedY]

plotY, plotYExtr = [], []
numberOfFratures = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

for _, clf in enumerate(clfs):
    print_to_file(clf)
    resultExtractions = []
    resultNoExtractions = []
    for i in numberOfFratures:
        xx, yy = createData(i)
        resultNoExtraction, balancedNoExtraction = makeExperiment(xx, yy, clf)
        resultNoExtractions.append(np.mean(resultNoExtraction))

        pca = PCA(n_components=np.min([800, round(i * 0.25)]))
        pca.fit(xx[:round(len(xx) * 0.8),:])
        extractedX = pca.transform(xx)

        resultExtraction, balancedExtraction = makeExperiment(extractedX, yy, clf)
        resultExtractions.append(np.mean(resultExtraction))

    print_to_file(clf)
    print_to_file(str(resultNoExtractions))
    print_to_file(str(resultExtractions))

    print(resultExtractions)
    maxAvg = np.argmax(resultExtractions)
    print('MAX', maxAvg)



# bez ekstrakcji
print_to_file('-----------------------BEZ EKSTRAKCJI-----------------------------')
for _, clf in enumerate(clfs):
    print_to_file(clf)
    for ix, x in enumerate(Xs):
        print_to_file(ix)
        result, balanceResult = makeExperiment(x[:, :100], ys[ix], clf)
        print_to_file(str(np.mean(result)))
        print_to_file(str(np.mean(balanceResult)))


# z ekstrakcja
print_to_file('-----------------------------EXTRACTED----------------------------')
for _, clf in enumerate(clfs):
    print_to_file(clf)
    for ix, x in enumerate(Xs):
        print_to_file(ix)
        pca = PCA(n_components=np.min([800, round(100 * 0.25)]))
        pca.fit(x[:round(len(x) * 0.8), :100])
        extractedX = pca.transform(x[:, :100])
        result, balanceResult = makeExperiment(extractedX, ys[ix], clf)
        print_to_file(str(np.mean(result)))
        print_to_file(str(np.mean(balanceResult)))
