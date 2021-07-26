<<<<<<< HEAD:project-keras.py
from keras.backend import dropout
from keras.layers.core import Dropout
import numpy as np
=======
>>>>>>> restructure:project_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
<<<<<<< HEAD:project-keras.py
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

def create_model(layers, activations):
    model = Sequential()
    model.add(Dense(layers[0], activation=activations[0]))
    model.add(Dense(layers[1], activation=activations[1]))
    model.add(Dense(layers[2], activation=activations[2]))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mse'])
    history = model.fit(X_train, y_train, epochs=11, verbose=0)
    model_metrics = model.evaluate(X_test, y_test)
    print('({l1}, {l2}, {l3}), ({a1}, {a2}, {a3})'.format(
        l1=layers[0], l2=layers[1], l3=layers[2], a1=activations[0], a2=activations[1], a3=activations[2]))
    print(model_metrics)
    plt.plot(history.history['accuracy'], label='accuracy ({l1}, {l2}, {l3}) ({a1}, {a2}, {a3})'.format(
        l1=layers[0], l2=layers[1], l3=layers[2], a1=activations[0], a2=activations[1], a3=activations[2]))
=======
from sklearn.model_selection import train_test_split
>>>>>>> restructure:project_preprocessing.py

allTrainingData = pd.read_csv('./train.csv')

X_train, X_test = train_test_split(allTrainingData, test_size=0.3, random_state=1)
y_train = X_train["label"]
<<<<<<< HEAD:project-keras.py
X_train = X_train.drop(labels="label", axis=1)
print(X_train)
y_test = X_test["label"]
X_test = X_test.drop(labels="label", axis=1)
=======
X_train.drop(labels="label", axis=1, inplace=True)

y_test = X_test["label"]
X_test.drop(labels="label", axis=1, inplace=True)
>>>>>>> restructure:project_preprocessing.py

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder()
encoder.fit(y_train.values.reshape(-1, 1))
y_train = encoder.transform(y_train.values.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.values.reshape(-1, 1)).toarray()
<<<<<<< HEAD:project-keras.py

create_model([784, 784, 10], ['relu', 'relu', 'softmax'])
create_model([784, 50, 10], ['relu', 'relu', 'softmax'])
create_model([784, 20, 10], ['relu', 'relu', 'softmax'])
create_model([784, 10, 10], ['relu', 'relu', 'softmax'])
plt.title('Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
plt.clf()

create_model([784, 784, 10], ['relu', 'sigmoid', 'softmax'])
create_model([784, 50, 10], ['relu', 'sigmoid', 'softmax'])
create_model([784, 20, 10], ['relu', 'sigmoid', 'softmax'])
create_model([784, 10, 10], ['relu', 'sigmoid', 'softmax'])
plt.title('Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
plt.clf()

create_model([784, 784, 10], ['relu', 'relu', 'sigmoid'])
create_model([784, 50, 10], ['relu', 'relu', 'sigmoid'])
create_model([784, 20, 10], ['relu', 'relu', 'sigmoid'])
create_model([784, 10, 10], ['relu', 'relu', 'sigmoid'])
plt.title('Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
plt.clf()
=======
>>>>>>> restructure:project_preprocessing.py
