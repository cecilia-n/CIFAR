import numpy as np
import pandas as pd
from numpy import array
from scipy.sparse.construct import rand
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

allTrainingData = pd.read_csv('./train.csv')
allTestData = pd.read_csv('./test.csv')

# training_labels = allTrainingData["label"]
# allTrainingData = allTrainingData.drop(labels="label", axis=1)
X_train, X_test = train_test_split(allTrainingData, test_size=0.3, random_state=1)
y_train = X_train["label"]
X_train.drop(labels="label", axis=1, inplace=True)

y_test = X_test["label"]
X_test.drop(labels="label", axis=1, inplace=True)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder()
encoder.fit(y_train.values.reshape(-1, 1))
y_train = encoder.transform(y_train.values.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

mlp = MLPClassifier(hidden_layer_sizes=(5), learning_rate="constant", 
                    learning_rate_init=0.01, max_iter=300, activation="relu", solver="adam", random_state=1)

mlp.fit(X_train, y_train)
# activation = ["identity", "logistic", "tanh", "relu"]
# solver = ["lbfgs", "sgd", "adam"]

# parameter_space =  {
#     'activation' : activation,
#     'solver' : solver
# }

# gridSearch = GridSearchCV(mlp, param_grid = parameter_space, n_jobs=-1, cv=5)
# gridSearch.fit(X_train, y_train)
# params = gridSearch.best_params_
# score = gridSearch.best_score_
# print(params)
# print(score)
print(classification_report(y_test, mlp.predict(X_test)))
