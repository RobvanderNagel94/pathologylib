from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


clfs = [
    {"name": 'LR',
     "estimator": LogisticRegression(),
     "params": {
         "C": [1000, 100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]
     }
     },

    {"name": 'SVC',
     "estimator": SVC(),
     "params": {
         "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
         "gamma": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
         "kernel": ['rbf']
     }
     },

    {"name": 'RF',
     "estimator": RandomForestClassifier(),
     "params": {
         "criterion": ['gini', 'entropy'],
         "bootstrap": [True, False],
         "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "max_features": ['auto'],
         "n_estimators": [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
         "random_state": [42]
     }
     },

    {"name": 'CART',
     "estimator": DecisionTreeClassifier(),
     "params": {
         "criterion": ['gini', 'entropy'],
         "splitter": ['best'],
         "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "min_samples_split": [2, 3, 4],
         "min_samples_leaf": [1, 2, 3, 4],
         "max_features": ['auto'],
         "random_state": [42]
     }
     },

    {"name": 'KNN',
     "estimator": KNeighborsClassifier(),
     "params": {
         "n_neighbors": [2, 4, 6, 8, 10],
         "weights": ['distance'],
         "algorithm": ['auto'],
         "p": [1, 2],
     }
     },
]

