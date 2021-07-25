from keras.layers import Dense
from keras.models import Sequential

import project_preprocessing
from project_preprocessing import X_train, y_train, X_test, y_test

model = Sequential()
model.add(Dense(784, activation='relu'))
model.add(Dense(784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mse'])
model.fit(X_train, y_train, epochs=11)
model_metrics = model.evaluate(X_test, y_test)
print(model_metrics)
