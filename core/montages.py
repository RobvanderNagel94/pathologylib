from core.constants import *


def reference_channels(dic: dict, montage_type: str = '1020') -> dict:
    """Re-reference channels according to a specified montage type.

    :param dic: Dictionary with keys being the channels and values the signal data (1020 system)
    :param montage_type: Type of montage to apply. Should be one of:
      'LPM', 'SLM', 'REF', 'G19', '1020'

    :return: Dictionary with re-referenced channels.
    """

    # Define channel mappings for longitudinal bipolar montage (e.g., double banana)
    if montage_type == 'LPM':

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
            re_ref_channels[pair_name] = dic[chan1] - dic[chan2]

        return re_ref_channels

    # Define channel mappings for small laplacian montage
    elif montage_type == 'SLM':

        channel_map = {

            L_T7: (T3, F7, C3, T5),
            L_P7: (T5, T3, P3, O1),
            L_FP1: (FP1, F7, FP2, F3),
            L_F3: (F3, FP1, F7, FZ, C3),
            L_C3: (C3, F3, T3, CZ, P3),
            L_P3: (P3, C3, T5, PZ, O1),
            L_O1: (O1, T5, P3, O2),
            L_FZ: (FZ, FP1, FP2, F3, F4, CZ),
            L_CZ: (CZ, C3, FZ, PZ, C4),
            L_PZ: (FZ, O2, P3, P4, O1, CZ),
            L_FP2: (FP2, FP1, F8, F4),
            L_F4: (F4, FP2, F8, FZ, C4),
            L_C4: (C4, F4, T4, CZ, P4),
            L_P4: (P4, C4, T6, PZ, O2),
            L_O2: (O2, P4, T6, O1),
            L_F7: (F8, FP2, F4, T4),
            L_T8: (T4, T6, C4, F8),
        }

        re_ref_channels = {}
        for pair_name, chan in channel_map.items():
            vals = dic[chan[0]] - sum(dic[ch] for ch in chan[1:]) / (len(chan) - 1)
            re_ref_channels[pair_name] = vals

        return re_ref_channels

    # Define channel mappings for referential montage
    elif montage_type == 'REF':
        re_ref_channels = {}
        for chan_name, val in dic.items():
            if chan_name == CZ:
                re_ref_channels["ref_" + chan_name] = val
            else:
                re_ref_channels["ref_" + chan_name] = val - dic[CZ]

        return re_ref_channels

    # Define channel mappings for source montage
    elif montage_type == 'G19':

        channel_map = {
            SD_FP2: (FP2, FZ, F4, F8),
            SD_FP1: (FP1, FZ, F3, F7),
            SD_F8: (F8, T4, C4, F4),
            SD_F7: (F7, T3, C3, F3),
            SD_F4: (F4, FZ, F8, C4),
            SD_F3: (F3, FZ, F7, C3),
            SD_T4: (T4, F8, C4, T6),
            SD_T3: (T3, F7, C3, T5),
            SD_C4: (C4, CZ, F4, P4, T4),
            SD_C3: (C3, CZ, F3, P3, T3),
            SD_T6: (T6, T4, P4, O2),
            SD_T5: (T5, T3, P3, O1),
            SD_P4: (P4, C4, PZ, T6, O2),
            SD_P3: (P3, C3, PZ, T5, O1),
            SD_O2: (O2, P4, PZ, T6),
            SD_O1: (O1, P3, PZ, T5),
            SD_FZ: (FZ, F3, F4, CZ),
            SD_CZ: (CZ, FZ, C4, C3, PZ),
            SD_PZ: (PZ, CZ, P4, P3, O2, O1),
        }

        re_ref_channels = {}
        for pair_name, chan in channel_map.items():
            vals = dic[chan[0]] - sum(dic[ch] for ch in chan[1:]) / (len(chan) - 1)
            re_ref_channels[pair_name] = vals

        return re_ref_channels

    # Define channel mappings for 1020 montage
    if montage_type == '1020':
        return dic
