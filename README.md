# ML_learning

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))


### working with classifier

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

#classify using classifier

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
# the below code upto print accuracy score wll remain same even your classifier is change only above two line will get change. 

my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print (predictions)

#to calculate the accuracy

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))

# trying  the same with different classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test)

print (predictions)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))
