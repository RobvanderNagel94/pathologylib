from statsmodels.stats.contingency_tables import mcnemar
from numpy import (ndarray, asarray)
from typing import Union


def McNemarsTest(y_true: Union[ndarray, list],
                y_pred_base: Union[ndarray, list],
                y_pred_comp: Union[ndarray, list]
                ) -> (float, float):
    """
    Compare model performances between two classifier models using the McNemar's test.

    :param y_true: Ground thruth of binary values, iterable of shape (n,).
    :param y_pred_base: Predicted binary values from the base model, iterable of shape (n,).
    :param y_pred_comp: Predicted binary values from the model to compare, iterable of shape (n,).

    :return: (p_value, test_statistic)

    Note:
    -------
    The test is applied to a 2x2 contingency table.
        H0 : Model errors are significantly different.
        H1 : Model errors are not significantly different.
    """

    y_true = asarray(y_true)
    y_pred_base = asarray(y_pred_base)
    y_pred_comp = asarray(y_pred_comp)

    if not len(y_true) == len(y_pred_base) == len(y_pred_comp):
        raise ValueError(f"Shape must be the same for all input arrays")

    counts_base = ['yes' if y_pred_base[i] == y_true[i] else 'no' for i in range(len(y_true))]
    counts_compare = ['yes' if y_pred_comp[i] == y_true[i] else 'no' for i in range(len(y_true))]

    yes_yes, yes_no, no_yes, no_no = 0, 0, 0, 0
    for i in range(len(counts_base)):
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'yes'):
            yes_yes += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'no'):
            no_no += 1
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'no'):
            yes_no += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'yes'):
            no_yes += 1

    contingency_table = [[yes_yes, yes_no],
                         [no_yes, no_no]]

    result = mcnemar(contingency_table, exact=False, correction=True)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors, failed to reject H0')
    else:
        print('Different proportions of errors, reject H0')

    return result.statistic, result.pvalue
