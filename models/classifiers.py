from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
from models.grid import estimators
from numpy import ndarray
from tqdm import tqdm


def run_grid_search(X_train: ndarray,
                    y_train: ndarray,
                    X_test: ndarray,
                    y_test: ndarray):
    """
    Searches the best classifier-specific parameters for a given binary classifier.

    :param X_train: Training set of features, 2D numpy array.
    :param y_train: Training set of labels, 1D numpy array.
    :param X_test: Testing set of features, 2D numpy array.
    :param y_test: Testing set of labels, 1D numpy array.

    :return: Tuple of optimal parameters, prediction results, and performance values.

    """

    # Ignore ConvergenceWarnings
    filterwarnings('ignore')

    # Scale feature inputs
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Initialize lists to store results
    specificity, sensitivity, accuracy = [], [], []
    y_preds, y_probas = [], []
    best_params = []

    # Loop over each classifier in the estimators list
    for clf in tqdm(estimators):

        model = GridSearchCV(estimator=clf["estimator"],
                             param_grid=clf["params"],
                             cv=8
                             ).fit(X_train_scaled, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Compute specificity, sensitivity, and accuracy
        specificity.append(tn / (tn + fp))
        sensitivity.append(tp / (tp + fn))
        accuracy.append((tp + tn) / (tp + fp + fn + tn))

        # Append predictions and prediction probabilities
        y_preds.append(model.predict(X_test_scaled))
        y_probas.append(model.predict_proba(X_test_scaled))

        # Append best parameters for each classifier
        best_params.append(model.best_params_)

    return best_params, (y_preds, y_probas), (specificity, sensitivity, accuracy)
