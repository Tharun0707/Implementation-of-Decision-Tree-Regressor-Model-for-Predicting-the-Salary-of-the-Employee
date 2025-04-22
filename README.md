# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features
2. Split data into training and testing sets
3. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features
4. Determine maximum depth of tree and other hyperparameters
5. Train your model -Fit model to training data -Calculate mean salary value for each subset
6. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
7. Tune hyperparameters -Experiment with different hyperparameters to improve performance
8. Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Tharun Sridhar
RegisterNumber: 212223230230 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
# Initial dataset:

![image](https://github.com/user-attachments/assets/af9b15f0-3a38-4bee-abee-b0451d8973d3)

# Data Info:

![image](https://github.com/user-attachments/assets/71159ee6-f54b-414a-9012-a28ec32cce2a)

# Optimization of null values:

![image](https://github.com/user-attachments/assets/6eba65a5-268b-4174-9c8c-8d4a6661d1f8)

# Converting string literals to numericl values using label encoder:

![image](https://github.com/user-attachments/assets/d77632cd-84fa-42c6-ba81-200fcb0296fa)

# Assigning x and y values:

![image](https://github.com/user-attachments/assets/f52a9ae3-f588-46cf-a0cd-00d063950900)

# Mean Squared Error:

![image](https://github.com/user-attachments/assets/6177888e-2757-4927-8d5c-b6349d0e4591)

# R2 (variance):

![image](https://github.com/user-attachments/assets/399108d7-3ea7-4738-9c72-2bbc0ffdbe40)

# Prediction:

![image](https://github.com/user-attachments/assets/dc0436ab-f5b5-41f0-a7f2-007f7e313ad3)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
