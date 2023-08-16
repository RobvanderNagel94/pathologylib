from models.classifiers import clfs

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y
from sklearn.metrics import confusion_matrix
from numpy import ndarray
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    logger.info(f"Running cross-validation for {clf['name']}...")

    search = GridSearchCV(estimator=clf["estimator"],
                          param_grid=clf["params"],
                          cv=cv,
                          scoring=scoring).fit(X_scaled, y)

    
    logger.info(f"Cross-validation completed for {clf['name']}.")

    return sc, search


def base_models(X, y, scoring='roc_auc', clfs: list = clfs, seed: int = 42) -> None:
    """
    Validates classifiers based on X and y data.

    :param clfs: list of dictionaries holding various classifiers and hyperparamaters
    :param X: Training set of features, of shape (n,m).
    :param y: Training set of labels, of shape (n,).
    :param scoring: Test metric.
    :param seed: Seed to be set to devide the dataset in train and test

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    for clf in clfs:
        sc, model = cross_validate(clf, X_train, y_train, scoring)
        X_test_scaled = sc.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        acc = (tp + tn) / (tp + fp + fn + tn)

        print(f"{clf['name']} Acc: {acc}, Spec: {spec}, Sens: {sens}")
