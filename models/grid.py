from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


estimators = [
    {"estimator": LogisticRegression(),
     "params": {
         "solver": ['newton-cg', 'lbfgs', 'liblinear'],
         "penalty": ['none', 'l1', 'l2', 'elasticnet'],
         "C": [1000, 100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]
                }
     },

    {"estimator": SVC(),
     "params": {
         "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
         "gamma": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
         "kernel": ['rbf']
                }
     },

    {"estimator": RandomForestClassifier(),
     "params": {
         "criterion": ['gini', 'entropy'],
         "bootstrap": [True, False],
         "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "max_features": ['sqrt', 'log2'],
         "n_estimators": [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
         "random_state": [42]
                }
     },
]