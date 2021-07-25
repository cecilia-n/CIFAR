from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import project_preprocessing
from project_preprocessing import X_train, y_train, X_test, y_test

mlp = MLPClassifier(hidden_layer_sizes=(5,), learning_rate="constant", 
                    learning_rate_init=0.01, max_iter=300, activation="relu", solver="adam", random_state=1)

mlp.fit(X_train, y_train)
print(classification_report(y_test, mlp.predict(X_test)))
print(mlp.score(X_test, y_test))
