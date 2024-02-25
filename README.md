# EXPERIMENT - 01 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

* Neural network regression models are a type of machine learning algorithm designed for predicting continuous numerical values. Comprising interconnected nodes organized in layers, neural networks can capture complex relationships within data. The input layer receives features, while hidden layers process information through weighted connections, and the output layer produces the regression prediction.
*  Training involves adjusting weights based on the model's error compared to actual values, optimizing performance.
*  Neural network regression models excel in handling nonlinear patterns and are widely used in various fields, such as finance, economics, and engineering, due to their ability to learn intricate relationships in data for accurate numeric predictions.

## NEURAL NETWORK MODEL
The neural network architecture comprises two hidden layers with ReLU activation functions, each having 5 and 3 neurons respectively, and a linear output layer with 1 neuron.

<img width="503" alt="EX1 NN MODEL" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/ecff89c1-93a7-48db-ac85-e5c2a0589408">

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Y SHAVEDHA
### Register Number: 212221230095
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('Dlexp1 ').sheet1

data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT':'float'})
dataset1 = dataset1.astype({'OUTPUT':'float'})
dataset1.head()

import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train1 = scaler.transform(X_train)

X_train1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')

X_test1 = scaler.transform(X_test)

model.evaluate(X_test1,y_test)

X_n1 = [[21]]
X_n1_1 = scaler.transform(X_n1)
model.predict(X_n1_1)

```
## Dataset Information
<img width="137" alt="image" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/a1c7a570-a6ba-40a4-819f-982fa57766fd">

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="397" alt="image" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/f2809941-2e93-4fad-bd2b-5bcdcb7a135f">

### Epoch Training
<img width="412" alt="image" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/eab700f4-016d-4915-84e3-c7eec39788ef">

### Test Data Root Mean Squared Error
<img width="373" alt="image" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/4f9374cb-89c0-4d10-8581-66e419ec0027">

### New Sample Data Prediction
<img width="380" alt="image" src="https://github.com/Shavedha/basic-nn-model/assets/93427376/15836cf4-d201-4495-8588-196723f44517">

## RESULT
Thus a Neural Network Regression model is executed successfully for the given inputs.
