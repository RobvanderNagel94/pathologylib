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

    pipe = Pipeline([('kbest', SelectKBest(method)), ('clf', clf)])

    Kfeatures = {'kbest__k': arange(1, X.shape[1])}
    search = GridSearchCV(pipe, Kfeatures, cv=8).fit(X, y)

    print('Best Mean Accuracy: %.3f' % search.best_score_)
    print('Best Config: %s' % search.best_params_)

    K = search.best_params_['kbest__k']
    features = SelectKBest(score_func=method, k=K).fit(X, y)

    dic = {idx: score for idx, score in enumerate(features.scores_)}

    return K, sorted(dic.items(), key=lambda x: x[1], reverse=True)

