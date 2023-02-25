from statsmodels.stats.contingency_tables import mcnemar
from numpy import ndarray


def McNemarTest(y_true: ndarray,
                y_pred_base: ndarray,
                y_pred_comp: ndarray
                ) -> (float, float):
    """
    Compare model performances between two classifier models using the McNemar's test.

    :param y_true: Ground thruth of binary values.
    :param y_pred_base: Predicted binary values from the base model.
    :param y_pred_comp: Predicted binary values from the model to compare.

    :return: (p_value, test_statistic)

    Note:
    -------
    The test is applied to a 2x2 contingency table.
        H0 : Model errors are significantly different.
        H1 : Model errors are not significantly different.
    """

    # Get counts for correct predictions of both models
    counts_base = ['yes' if y_pred_base[i] == y_true[i] else 'no' for i in range(len(y_true))]
    counts_compare = ['yes' if y_pred_comp[i] == y_true[i] else 'no' for i in range(len(y_true))]

    # Initialize the variables for contingency table
    yes_yes, yes_no, no_yes, no_no = 0, 0, 0, 0

    # Count number of correct and incorrect predictions
    for i in range(len(counts_base)):
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'yes'):
            yes_yes += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'no'):
            no_no += 1
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'no'):
            yes_no += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'yes'):
            no_yes += 1

    # Define 2x2 contingency table
    contingency_table = [[yes_yes, yes_no],
                         [no_yes, no_no]]

    # Perform McNemar's test with continuity correction
    result = mcnemar(contingency_table, exact=False, correction=True)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    # Compute the significance of H0 and H1
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors, failed to reject H0')
    else:
        print('Different proportions of errors, reject H0')

    return result.statistic, result.pvalue
