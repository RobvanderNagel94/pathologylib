from sklearn.metrics import roc_auc_score
from seaborn import set_style
from numpy import ndarray

from matplotlib.pyplot import (
    plot,
    rcParams,
    rc,
    xlabel,
    ylabel,
    xlim,
    ylim,
    legend,
    grid,
    show)


def plot_roc_curve(y_preds: ndarray,
                   y_probas: ndarray,
                   p_values: ndarray,
                   names: ndarray
                   ) -> None:
    """
    Plot ROC curves and show AUC, p-value for each curve.

    :param y_preds: Binary ground truth (correct) target values.
    :param y_probas: Predicted target probabilities for positive class.
    :param p_values: P-values for each curve.
    :param names: Names for each curve.
    """

    set_style("whitegrid")
    rcParams["figure.figsize"] = (10, 8)
    rcParams['legend.loc'] = "best"

    rc('font', size=13)  
    rc('axes', titlesize=16)  
    rc('axes', labelsize=13)  
    rc('xtick', labelsize=13)  
    rc('ytick', labelsize=13)  
    rc('legend', fontsize=13)  
    rc('figure', titlesize=16)  

    tprs, fprs = [], []
    for i in range(len(y_preds)):
        fpr, tpr, _ = metrics.roc_curve(y_preds[i], y_probas[i])
        auc = metrics.roc_auc_score(y_preds[i], y_probas[i])
        tprs.append(tpr)
        fprs.append(fpr)
        plot(fpr,tpr,"o-",label=f"{names[i]}: auc={round(auc, 3)}, p={round(p_values[i], 3)}",linewidth=1,markersize=5)

    plot([0, 1], [0, 1], 'k--', linewidth=1)
    xlim([0.0, 1.0])
    ylim([0.0, 1.05])
    xlabel('False Positive Rate (FPR)')
    ylabel('True Positive Rate (TPR)')
    legend(loc="lower right")
    grid()
    show()
