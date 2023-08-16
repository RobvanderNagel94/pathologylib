from core.montages import MONTAGE_1020

from matplotlib.pyplot import (subplots, show)
from mne import (create_info, viz)
from numpy import (ndarray, char)


def plot_topograph(feat_importance_02: ndarray,
                   feat_importance_24: ndarray,
                   feat_importance_48: ndarray,
                   feat_importance_813: ndarray,
                   feat_importance_1318: ndarray,
                   feat_importance_1824: ndarray,
                   feat_importance_2430: ndarray,
                   feat_importance_3050: ndarray,
                   montage=MONTAGE_1020) -> None:
    """
    Plot topographs of feature importance generated from a random forest classifier.

    :param feat_importance_02: Importance values for frequency band 0-2Hz.
    :param feat_importance_24: Importance values for frequency band 2-4Hz.
    :param feat_importance_48: Importance values for frequency band 4-8Hz.
    :param feat_importance_813: Importance values for frequency band 8-13Hz.
    :param feat_importance_1318: Importance values for frequency band 13-18Hz.
    :param feat_importance_1824: Importance values for frequency band 18-24Hz.
    :param feat_importance_2430: Importance values for frequency band 24-30Hz.
    :param feat_importance_3050: Importance values for frequency band 30-50Hz.
    :param montage: Type of montage used to visualize.

    Notes
    -----
    Each array consist of feature importance values at different frequency bands.
    These values can be generated using the following code snippet:

        clf = RandomForestClassifier().fit(X,y)
        print(clf.feature_importances_)

    """

    MONTAGE = char.upper(montage).tolist()

    info = create_info(ch_names=MONTAGE, sfreq=250, ch_types='eeg')
    info.set_montage('standard_1020')

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = subplots(2, 4, figsize=(15, 7))

    im, cm = viz.plot_topomap(feat_importance_02, info, axes=ax1, show=False)
    im, cm = viz.plot_topomap(feat_importance_24, info, axes=ax2, show=False)
    im, cm = viz.plot_topomap(feat_importance_48, info, axes=ax3, show=False)
    im, cm = viz.plot_topomap(feat_importance_813, info, axes=ax4, show=False)
    im, cm = viz.plot_topomap(feat_importance_1318, info, axes=ax5, show=False)
    im, cm = viz.plot_topomap(feat_importance_1824, info, axes=ax6, show=False)
    im, cm = viz.plot_topomap(feat_importance_2430, info, axes=ax7, show=False)
    im, cm = viz.plot_topomap(feat_importance_3050, info, axes=ax8, show=False)

    ax_x_start = 0.95
    ax_x_width = 0.02
    ax_y_start = 0.1
    ax_y_height = 0.8

    ax1.set_title('0-2Hz')
    ax2.set_title('2-4Hz')
    ax3.set_title('4-8Hz')
    ax4.set_title('8-13Hz')
    ax5.set_title('13-18Hz')
    ax6.set_title('18-24Hz')
    ax7.set_title('24-30Hz')
    ax8.set_title('30-50Hz')

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)

    show()
