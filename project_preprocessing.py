import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

allTrainingData = pd.read_csv('./train.csv')
allTestData = pd.read_csv('./test.csv')

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
