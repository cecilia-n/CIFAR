from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import project_preprocessing
from project_preprocessing import X_train, y_train, X_test, y_test

mlp = MLPClassifier(hidden_layer_sizes=(5), learning_rate="constant", 
                    learning_rate_init=0.01, max_iter=300, activation="relu", solver="adam", random_state=1)

activation = ["identity", "logistic", "tanh", "relu"]
solver = ["lbfgs", "sgd", "adam"]

parameter_space =  {
    'activation' : activation,
    'solver' : solver
}

gridSearch = GridSearchCV(mlp, param_grid = parameter_space, n_jobs=-1, cv=5)
gridSearch.fit(X_train, y_train)
params = gridSearch.best_params_
score = gridSearch.best_score_
print(params)
print(score)
