from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
housing = fetch_california_housing()

# Features / labels
X = housing.data
y = X.T[3]

# Model
model = LinearRegression()

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions : ", predictions)

r_squared = model.score(X_train, y_train)
print("Coefficient of determination : ", r_squared)

plt.scatter(X.T[2], X.T[3])
plt.xlabel('AveRooms')
plt.ylabel('AveBedrms')
plt.show()
