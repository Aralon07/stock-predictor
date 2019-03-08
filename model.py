from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

data = np.genfromtxt('feature-extraction/features-without-date.csv', delimiter = ',', dtype = float)

X_train, y_train, X_test, y_test = data[:600, :-1], data[:600, -1], data[600:800, :-1], data[600:800, -1]

model = LogisticRegression(solver = 'lbfgs', max_iter = 1000)

model = model.fit(X_train, y_train)

train_predicted = model.predict(X_train)

predicted = model.predict(X_test)

print("Accuracy For LogisticRegression for train: ", metrics.accuracy_score(y_train, train_predicted))
print("Accuracy For LogisticRegression for test: ", metrics.accuracy_score(y_test, predicted))

# print(metrics.confusion_matrix(y_train, train_predicted))

# print(predicted)

# print(metrics.confusion_matrix(y_test, predicted))

model2 = SGDClassifier()

model2 = model2.fit(X_train, y_train)

train_predicted = model2.predict(X_train)

predicted = model2.predict(X_test)

print("Accuracy For SGDClassifier for train: ", metrics.accuracy_score(y_train, train_predicted))
print("Accuracy For SGDClassifier for test: ", metrics.accuracy_score(y_test, predicted))


# print(metrics.confusion_matrix(y_train, train_predicted))

print(predicted)

# print(metrics.confusion_matrix(y_test, predicted))
