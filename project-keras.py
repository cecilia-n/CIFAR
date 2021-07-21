import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.models import Sequential

allTrainingData = pd.read_csv('./train.csv')
allTestData = pd.read_csv('./test.csv')

# training_labels = allTrainingData["label"]
# allTrainingData = allTrainingData.drop(labels="label", axis=1)
X_train, X_test = train_test_split(allTrainingData, test_size=0.3, random_state=1)
y_train = X_train["label"]
X_train.drop(labels="label", axis=1)

y_test = X_test["label"]
X_test.drop(labels="label", axis=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder()
encoder.fit(y_train.values.reshape(-1, 1))
y_train = encoder.transform(y_train.values.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

model = Sequential()
model.add(Dense(784, activation='relu'))
model.add(Dense(784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mse'])
model.fit(X_train, y_train, epochs=11)
model_metrics = model.evaluate(X_test, y_test)
print(model_metrics)
