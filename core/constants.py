import numpy as np

# meta data
SFREQ = '_sfreq'
ANNOT = '_annot'
GENDER = '_gender'
AGE = '_age'
ID = '_id'

# 10-20 international system
FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ = 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4', 'fz', 'cz'
PZ, F3, C3, P3, FP1, F7, T3, T5, O1 = 'pz', 'f3', 'c3', 'p3', 'fp1', 'f7', 't3', 't5', 'o1'
MONTAGE_1020 = np.array([FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1])

# G19
SD_FP2, SD_F8, SD_T4, SD_T6, SD_O2, SD_F4, SD_C4, SD_P4, SD_FZ, SD_CZ = 'sd_fp2', 'sd_f8', 'sd_t4', 'sd_t6', 'sd_o2', 'sd_f4', 'sd_c4', 'sd_p4', 'sd_fz', 'sd_cz'
SD_PZ, SD_F3, SD_C3, SD_P3, SD_FP1, SD_F7, SD_T3, SD_T5, SD_O1 = 'sd_pz', 'sd_f3', 'sd_c3', 'sd_p3', 'sd_fp1', 'sd_f7', 'sd_t3', 'sd_t5', 'sd_o1'
MONTAGE_G19 = np.array([SD_FP2, SD_FP1, SD_F8, SD_F7, SD_F4, SD_F3, SD_T4, SD_T3,
                        SD_C4, SD_C3, SD_T6, SD_T5, SD_P4, SD_P3, SD_O2, SD_O1,SD_FZ, SD_CZ, SD_PZ])

# laplacian montage
L_T7, L_P7, L_FP1, L_F3, L_C3, L_P3, L_O1, L_FZ = 'l_t7', 'l_p7', 'l_fp1', 'l_f3', 'l_c3', 'l_p3', 'l_o1', 'l_fz'
L_CZ, L_PZ, L_FP2, L_F4, L_C4, L_P4, L_O2, L_F7, L_T8 = 'l_cz', 'l_pz', 'l_fp2', 'l_f4', 'l_c4', 'l_p4', 'l_o2', 'l_f7', 'l_t8'
MONTAGE_LAP = np.array([L_T7, L_P7, L_FP1, L_F3, L_C3, L_P3, L_O1, L_FZ, L_CZ, L_PZ, L_FP2, L_F4, L_C4, L_P4, L_O2, L_F7, L_T8])


# longitudinal bipolar montage
FP2_F8, F8_T4, T4_T6, T6_O2, FP2_F4, F4_C4, C4_P4, P4_O2 = 'fp2-f8', 'f8-t4', 't4-t6', 't6-o2', 'fp2-f4', 'f4-c4', 'c4-p4', 'p4-o2 '
FP1_F3, F3_C3, C3_P3, P3_O1, FP1_F7, F7_T3, T3_T5, T5_O1, FZ_CZ, CZ_PZ = 'fp1-f3', 'f3-c3', 'c3-p3', 'p3-o1', 'fp1-f7', 'f7-t3', 't3-t5', 't5-o1', 'fz-cz', 'cz-pz '
MONTAGE_LB = np.array([FP2_F8, F8_T4, T4_T6, T6_O2, FP2_F4, F4_C4, C4_P4, P4_O2, FP1_F3,
                       F3_C3, C3_P3, P3_O1, FP1_F7, F7_T3, T3_T5, T5_O1, FZ_CZ, CZ_PZ])
