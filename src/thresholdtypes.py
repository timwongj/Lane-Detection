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
    ADAPT_MEAN = 9,
    ADAPT_GAUSS = 10,
    OTSU = 11,
    OTSU_GAUSS = 12,
    HLS = 13,
    HSV = 14