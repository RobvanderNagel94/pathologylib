from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import (subplots, show)
from pandas import Series
from numpy import ndarray


def random_forest_feature_importance(X_train: ndarray,
                                     y_train: ndarray,
                                     X_test: ndarray,
                                     y_test: ndarray,
                                     feature_names: ndarray,
                                     cv: int = 8,
                                     criterion: str = 'gini',
                                     bootstrap: bool = True,
                                     max_depth: int = 5,
                                     max_features: str = 'sqrt',
                                     n_estimators: int = 3,
                                     random_state: int = 34,
                                     show_importance: bool = True) -> Series:
    """
    Quantify feature importance based on permutation feature importance.

    :param X_train: Training set of features, 2D numpy array.
    :param y_train: Training set of labels, 1D numpy array.
    :param X_test: Testing set of features, 2D numpy array.
    :param y_test: Testing set of labels, 1D numpy array.
    :param feature_names: Feature names for plotting and identifying important features.
    :param cv: Iterations for cross-validation.
    :param criterion: Splitting criterion for classifier.
    :param bootstrap: Whether to use bootstrap samples in the classifier.
    :param max_depth: Maximum depth of the tree in the classifier.
    :param max_features: Maximum number of features to use in each split of the classifier.
    :param n_estimators: Number of trees in the classifier.
    :param random_state: Random seed for reproducibility.
    :param show_importance: Whether to display feature importances in a bar plot.

    :return: Series of feature importances with feature names as index.

    References
    ----------
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    """

    # Scale features
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Fit classifier
    clf = RandomForestClassifier(criterion=criterion,
                                 bootstrap=bootstrap,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 n_estimators=n_estimators,
                                 random_state=random_state)

    model = GridSearchCV(estimator=clf, cv=cv).fit(X_train_scaled, y_train)

    # Fit permutation importance
    result = permutation_importance(model,
                                    X_test_scaled,
                                    y_test,
                                    n_repeats=10,
                                    random_state=random_state,
                                    n_jobs=2)

    feature_importance = Series(result.importances_mean, index=feature_names)

    # Show feature importance
    if show_importance:
        fig, ax = subplots()
        feature_importance.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using full permutation")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        show()
        return feature_importance

    return feature_importance

