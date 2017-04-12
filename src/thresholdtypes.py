from enum import IntEnum


class ThresholdTypes(IntEnum):
    COMBINED = 0,
    COMBINED_TOL = 1,
    MAG = 2,
    COLOR = 3,
    COLOR_TOL = 4,
    ABS_SOB_X = 5,
    ABS_SOB_Y = 6,
    ABS_SOB_X_TOL = 7,
    ABS_SOB_Y_TOL = 8,
    OTSU = 9,
    OTSU_GAUSS = 10,
    HLS = 11,
    HSV = 12,
    ADAPT_MEAN = 13,
    ADAPT_GAUSS = 14,