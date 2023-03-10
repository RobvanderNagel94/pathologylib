from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from numpy import (ndarray, arange)
from typing import Union


def select_k_best_features(X: Union[ndarray, list],
                           y: Union[ndarray, list],
                           clf: any = LogisticRegression(),
                           method: any = f_classif) -> (int, dict):
    """
    Iterative feature selection procedure to find the most optimal set of features.

    :param X: Training set of features, iterable of shape (n, m).
    :param y: Training set of labels, iterable of shape (n,).
    :param clf: Binary classifier of sklearn.
    :param method: Method to select features (f-test or mutual information).

    :return: (K, sorted dictionary of features and their importance (descending order))

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    """

    # Create pipeline
    pipe = Pipeline([('kbest', SelectKBest(method)), ('clf', clf)])

    # Define grid with the range of features to be selected
    Kfeatures = {'kbest__k': arange(1, X.shape[1])}
    search = GridSearchCV(pipe, Kfeatures, cv=8).fit(X, y)

    # Show best mean accuracy and corresponding configuration
    print('Best Mean Accuracy: %.3f' % search.best_score_)
    print('Best Config: %s' % search.best_params_)

    # Retrieve K features resulting in the best mean accuracy
    K = search.best_params_['kbest__k']

    # Select K features
    features = SelectKBest(score_func=method, k=K).fit(X, y)

    # Retrieve feature importance scores
    dic = {idx: score for idx, score in enumerate(features.scores_)}

    # Return K, sorted dictionary by feature importance in descending order
    return K, sorted(dic.items(), key=lambda x: x[1], reverse=True)

