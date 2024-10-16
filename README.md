# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required
2. Read the dataset.
3. Define X and Y array. 
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YOGESH. V
RegisterNumber:  212223230250
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1) 
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:
![Screenshot 2024-10-16 210946](https://github.com/user-attachments/assets/d205a318-f104-4c6d-8b95-63af53ea66b3)
![Screenshot 2024-10-16 211001](https://github.com/user-attachments/assets/3141e12e-f325-4bcc-a64e-850e1dc15e31)
![Screenshot 2024-10-16 211015](https://github.com/user-attachments/assets/bce71534-9a32-4f87-bd8c-9824f99a9317)
![Screenshot 2024-10-16 211026](https://github.com/user-attachments/assets/0844d002-4448-47a9-97c5-a39aa0d5191d)
![Screenshot 2024-10-16 211043](https://github.com/user-attachments/assets/b43dd063-1de1-4463-89e9-a6353fb70e3d)
![Screenshot 2024-10-16 211057](https://github.com/user-attachments/assets/7a1bd3f0-94ad-4e4c-b8cd-8582ce9e9dc8)
![Screenshot 2024-10-16 211123](https://github.com/user-attachments/assets/97de55db-f714-49aa-ac1b-03334d40a243)
![Screenshot 2024-10-16 211138](https://github.com/user-attachments/assets/c86f1aa7-2c94-4234-9337-9e74d05cdecd)
![Screenshot 2024-10-16 211150](https://github.com/user-attachments/assets/f399e422-b60d-4a98-b659-0b7c575bbcd4)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

