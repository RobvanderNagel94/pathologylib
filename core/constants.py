from numpy import array

FS = '_fs'
ANNOT = '_annot'
GENDER = '_gender'
AGE = '_age'
ID = '_id'

FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, T7 = 'fp2', 'f8', 't4', 't6', 'o2', 'f4', 'c4', 'p4', 'fz', 'cz', 't7'
PZ, F3, C3, P3, FP1, F7, T3, T5, O1, P7, T8 = 'pz', 'f3', 'c3', 'p3', 'fp1', 'f7', 't3', 't5', 'o1', 'p7', 't8'
FP2_F8, F8_T4, T4_T6, T6_O2, FP2_F4, F4_C4, C4_P4, P4_O2 = 'fp2-f8', 'f8-t4', 't4-t6', 't6-o2', 'fp2-f4', 'f4-c4', 'c4-p4', 'p4-o2 '
FP1_F3, F3_C3, C3_P3, P3_O1, FP1_F7, F7_T3, T3_T5, T5_O1, FZ_CZ, CZ_PZ = 'fp1-f3', 'f3-c3', 'c3-p3', 'p3-o1', 'fp1-f7', 'f7-t3', 't3-t5', 't5-o1', 'fz-cz', 'cz-pz '

# source reference montage
MONTAGE_1020 = array([FP2, F8, T4, T6, O2, F4, C4, P4, FZ, CZ, PZ, F3, C3, P3, FP1, F7, T3, T5, O1])
# small laplacian montage
MONTAGE_SLAP = array([T7, P7, FP1, F3, C3, P3, O1, FZ, CZ, PZ, FP2, F4, C4, P4, O2, F7, T8])
# longitudinal bipolar montage
MONTAGE_LTB = array([FP2_F8, F8_T4, T4_T6, T6_O2, FP2_F4, F4_C4, C4_P4, P4_O2, FP1_F3, F3_C3, C3_P3, P3_O1, FP1_F7, F7_T3, T3_T5, T5_O1, FZ_CZ, CZ_PZ])




