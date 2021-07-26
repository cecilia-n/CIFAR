from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import project_preprocessing
from project_preprocessing import X_train, y_train, X_test, y_test

mlp = MLPClassifier(hidden_layer_sizes=(400,200,100), learning_rate="constant", 
                    learning_rate_init=0.001, max_iter=100, activation="relu", solver="adam", random_state=1)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
print(classification_report(y_test, mlp.predict(X_test)))
