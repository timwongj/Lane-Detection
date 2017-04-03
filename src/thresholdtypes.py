from enum import IntEnum


class ThresholdTypes(IntEnum):
    COLOR = 0,
    COLOR_TOL = 1,
    MAG = 2,
    COMBINED = 3,
    COMBINED_TOL = 4,
    ADAPT_MEAN = 5,
    ADAPT_GAUSS = 6,
    OTSU = 7,
    OTSU_GAUSS = 8,
    ABS_SOB_X = 9,
    ABS_SOB_Y = 10,
    ABS_SOB_X_TOL = 11,
    ABS_SOB_Y_TOL = 12,
    HLS = 13,
    HSV = 14
