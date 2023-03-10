from models.grid import clfs

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y
from sklearn.metrics import confusion_matrix
from numpy import ndarray


def cross_validate(clf: dict, X: ndarray, y: ndarray, scoring: str, cv: int = 8):
    """
    Searches the best classifier-specific parameters for a given binary classifier.

    :param clf: Dictionaries holding the classifier and hyperparamaters
    :param X: Training set of features, of shape (n,m).
    :param y: Training set of labels, of shape (n,).
    :param scoring: Test metric.
    :param cv: Value for cross validation.

    :return: scaler, GridSearchCV object
    """

    # Check X and y inputs
    X, y = check_X_y(X, y,
                     accept_sparse=False,
                     accept_large_sparse=False,
                     dtype='numeric',
                     force_all_finite=True,
                     ensure_2d=True,
                     allow_nd=False,
                     multi_output=False,
                     ensure_min_samples=1,
                     ensure_min_features=1,
                     y_numeric=True)

    # Scale feature inputs
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    # Apply gridsearch of parameters
    search = GridSearchCV(estimator=clf["estimator"],
                          param_grid=clf["params"],
                          cv=cv,
                          scoring=scoring)

    # return scaler and cv_object
    return sc, search.fit(X_scaled, y)


def base_models(X, y, scoring='roc_auc', clfs: list = clfs, seed: int = 42) -> None:
    """
    Validates classifiers based on X and y data.

    :param clfs: list of dictionaries holding various classifiers and hyperparamaters
    :param X: Training set of features, of shape (n,m).
    :param y: Training set of labels, of shape (n,).
    :param scoring: Test metric.
    :param seed: Seed to be set to devide the dataset in train and test

    """

    # Test dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    for clf in clfs:
        # Cross validate based on train (e.g., train and validation)
        sc, model = cross_validate(clf, X_train, y_train, scoring)
        X_test_scaled = sc.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # Compute specificity, sensitivity, and accuracy
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        acc = (tp + tn) / (tp + fp + fn + tn)

        print(f"{clf['name']} Acc: {acc}, Spec: {spec}, Sens: {sens}")
