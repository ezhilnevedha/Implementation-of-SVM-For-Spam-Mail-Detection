# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.
## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Ezhil Nevedha K
RegisterNumber: 212223230055
*/
```
```python
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows=1252')
data.head()
```
![image](https://github.com/user-attachments/assets/61a4f022-1de3-4ecc-897f-ea85b03ebb7a)
```python
data.tail()
```
![image](https://github.com/user-attachments/assets/198557f0-2d3c-4819-a0cb-6b71f1644143)
```python
data.info()
```
![image](https://github.com/user-attachments/assets/a9a493d6-4ce7-4baf-9895-28c04a033672)
```python
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/eafbbfd7-e359-43f1-a0ac-9da49f2d4c3b)
```python
x=data["v2"].values
y=data["v1"].values
y.shape
```
![image](https://github.com/user-attachments/assets/308cd4bb-392c-4181-9976-9cd4b58e229c)
```python
x.shape
```
![image](https://github.com/user-attachments/assets/8bf88a26-0b95-445a-8f2b-c34e96033d93)
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```
![image](https://github.com/user-attachments/assets/64b454c9-4421-440d-a31d-a1e9092f6572)
```python
y_train.shape
```
![image](https://github.com/user-attachments/assets/0cf15b00-5b93-4d3b-bda8-01d47b5a2739)
```python
x_test.shape
```
![image](https://github.com/user-attachments/assets/107fecd7-2ea2-4902-8263-dbbfb553a7ae)
```python
y_test.shape
```
![image](https://github.com/user-attachments/assets/2d24c96c-29ca-446b-a8d2-83e79233cb94)
```python
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```python
x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)
```
```python
x_train.shape
```
![image](https://github.com/user-attachments/assets/945acd6f-7791-4ce7-bd87-5a1b5faae74c)
```python
type(x_train)
```
![image](https://github.com/user-attachments/assets/81f12100-08f5-4dda-8eda-9445d7dae32f)
```python
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/8015bef1-08c0-45fc-a2c7-ad3c9062d265)
```python
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/38ad238b-84e8-4047-83b7-be49c7fd7d3f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
