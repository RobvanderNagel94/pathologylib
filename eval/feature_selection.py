from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from numpy import (ndarray, arange)


def select_k_best_features(X: ndarray,
                           y: ndarray,
                           clf: any = LogisticRegression(),
                           method: any = f_classif) -> (int, dict):
    """
    Iterative feature selection procedure to find the most optimal set of features.

    :param X: Training set of features, 2D numpy array.
    :param y: Training set of labels, 1D numpy array.
    :param clf: Binary classifier of sklearn.
    :param method: Method to select features (f-test or mutual information).
    :param show_importance: Whether to display sorted feature importance of K selected features.

    :return: (K, sorted dictionary of features and their importance (descending order))

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    """

    # Create pipeline with SelectKBest and clf
    pipe = Pipeline([('kbest', SelectKBest(method)), ('clf', clf)])

    # Define grid with the range of features to be selected
    Kfeatures = {'kbest__k': arange(1, X.shape[1])}
    search = GridSearchCV(pipe, Kfeatures, cv=8).fit(X, y)

    # Print best mean accuracy and the corresponding configuration
    print('Best Mean Accuracy: %.3f' % search.best_score_)
    print('Best Config: %s' % search.best_params_)

    # Retrieve K features with the best mean accuracy
    K = search.best_params_['kbest__k']

    # Select K best features based on the method
    features = SelectKBest(score_func=method, k=K).fit(X, y)

    # Retrieve feature importance scores
    dic = {idx: score for idx, score in enumerate(features.scores_)}

    # Return K features, sorted dictionary by feature importance in descending order
    return K, sorted(dic.items(), key=lambda x: x[1], reverse=True)

