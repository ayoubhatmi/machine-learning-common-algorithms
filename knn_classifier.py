import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import car dataset
data = pd.read_csv('datasets/car.data')

# Features / Labels
X = data[['buying', 'maint', 'safety']].values
y = data[['class']]
X = np.array(X)

# Converting data
# X
for i in range(len(X[0])):
    X[:, i] = LabelEncoder().fit_transform(X[:, i])

# y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)

# Model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

# Model evaluation
accuracy = metrics.accuracy_score(y_test, predictions)
print("predictions:", predictions)
print("accuracy: ", accuracy)

a = 1727
print("actual value ", y[a])
print("predicted value", knn.predict(X)[a])
