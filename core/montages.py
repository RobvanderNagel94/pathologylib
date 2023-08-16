from core.constants import *


def reference_channels(eeg_dic: dict, type: str = '1020') -> dict:
    """
    Reference channels according to a specified montage type.

    :param eeg_dic: Dictionary with keys being the channels and values the signal data (1020 system)
    :param type: Type of montage to apply. Should be one of:
      'longitudinal_bipolar', 'small_laplacian', 'reference', 'source', '1020'

    :return: Dictionary with re-referenced channels.

    Notes
    ----------
    The input dic must satisfy the output of the 1020 system with 19 channels:
    MONTAGE_1020 = [FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1]
    """

    if type == 'longitudinal_bipolar':

        channel_map = {

            FP2_F8: (FP2, F8),
            F8_T4: (F8, T4),
            T4_T6: (T4, T6),
            T6_O2: (T6, O2),
            FP2_F4: (FP2, F4),
            F4_C4: (F4, C4),
            C4_P4: (C4, P4),
            P4_O2: (P4, O2),
            FP1_F3: (FP1, F3),
            F3_C3: (F3, C3),
            C3_P3: (C3, P3),
            P3_O1: (P3, O1),
            FP1_F7: (FP1, F7),
            F7_T3: (F7, T3),
            T3_T5: (T3, T5),
            T5_O1: (T5, O1),
            FZ_CZ: (FZ, CZ),
            CZ_PZ: (CZ, PZ),
        }

        re_ref_channels = {}
        for pair_name, (chan1, chan2) in channel_map.items():
            re_ref_channels[pair_name] = eeg_dic[chan1] - eeg_dic[chan2]

        return re_ref_channels


    elif type == 'small_laplacian':

        channel_map = {

            T7: (T3, F7, C3, T5),
            P7: (T5, T3, P3, O1),
            FP1: (FP1, F7, FP2, F3),
            F3: (F3, FP1, F7, FZ, C3),
            C3: (C3, F3, T3, CZ, P3),
            P3: (P3, C3, T5, PZ, O1),
            O1: (O1, T5, P3, O2),
            FZ: (FZ, FP1, FP2, F3, F4, CZ),
            CZ: (CZ, C3, FZ, PZ, C4),
            PZ: (FZ, O2, P3, P4, O1, CZ),
            FP2: (FP2, FP1, F8, F4),
            F4: (F4, FP2, F8, FZ, C4),
            C4: (C4, F4, T4, CZ, P4),
            P4: (P4, C4, T6, PZ, O2),
            O2: (O2, P4, T6, O1),
            F7: (F8, FP2, F4, T4),
            T8: (T4, T6, C4, F8),
        }

        re_ref_channels = {}
        for pair_name, chan in channel_map.items():
            vals = eeg_dic[chan[0]] - sum(eeg_dic[ch] for ch in chan[1:]) / (len(chan) - 1)
            re_ref_channels[pair_name] = vals

        return re_ref_channels


    elif type == 'reference':
        re_ref_channels = {}
        for chan_name, val in eeg_dic.items():
            if chan_name == CZ:
                re_ref_channels[chan_name] = val
            else:
                re_ref_channels[chan_name] = val - eeg_dic[CZ]

        return re_ref_channels


    elif type == 'source':

        channel_map = {

            FP2: (FP2, FZ, F4, F8),
            F8: (F8, T4, C4, F4),
            T4: (T4, F8, C4, T6),
            T6: (T6, T4, P4, O2),
            O2: (O2, P4, PZ, T6),
            F4: (F4, FZ, F8, C4),
            C4: (C4, CZ, F4, P4, T4),
            P4: (P4, C4, PZ, T6, O2),
            FZ: (FZ, F3, F4, CZ),
            CZ: (CZ, FZ, C4, C3, PZ),
            PZ: (PZ, CZ, P4, P3, O2, O1),
            F3: (F3, FZ, F7, C3),
            C3: (C3, CZ, F3, P3, T3),
            P3: (P3, C3, PZ, T5, O1),
            FP1: (FP1, FZ, F3, F7),
            F7: (F7, T3, C3, F3),
            T3: (T3, F7, C3, T5),
            T5: (T5, T3, P3, O1),
            O1: (O1, P3, PZ, T5),
        }

        re_ref_channels = {}
        for pair_name, chan in channel_map.items():
            vals = eeg_dic[chan[0]] - sum(eeg_dic[ch] for ch in chan[1:]) / (len(chan) - 1)
            re_ref_channels[pair_name] = vals

        return re_ref_channels


    if type == '1020':
        return eeg_dic
