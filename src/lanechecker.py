import cv2
import math


class Lanechecker:
    """
    This class checks if two lanes are angled correctly. It does not check
    if the lanes are in the correct position. First it calculates
    the lanes' angles, then it checks to see if they're within
    a certain range, and finally it displays the results.

    How to use: Call check_lanes and pass the required parameters, specified
    below.

    Example:
        from lanechecker import Lanechecker
        lanechecker = Lanechecker()
        lanechecker.check_lanes(bot_left_x, top_left_x, bot_right_x,
                                top_right_x, bot_y, top_y, img)
    """
    # constructor
    def __init__(self):
        self.valid = None
        self.leftAngle = None
        self.rightAngle = None
        self.leftCheck = False
        self.rightCheck = False

    def calculate_angles(self, bot_left_x, top_left_x, bot_right_x,
                         top_right_x, bot_y, top_y):
        """
        Use trigonometry to calculate the two inner angles of the lanes.

        :param bot_left_x: bottom left x coordinate
        :param top_left_x: top left x coordinate
        :param bot_right_x: bottom right x coordinate
        :param top_right_x: top right x coordinate
        :param bot_y: bottom of both lines
        :param top_y: top of both lines
        :return: nothing
        """

        # calculate left angle

        # if lane is facing towards center
        if top_left_x > bot_left_x:
            self.leftAngle = math.atan(
                (bot_y - top_y) / (top_left_x - bot_left_x)
            )
        # if lane is facing away from center
        else:
            self.leftAngle = math.pi - math.atan(
                (bot_y - top_y) / (bot_left_x - top_left_x)
            )

        # calculate right angle

        # if lane is facing towards center
        if top_right_x < bot_right_x:
            self.rightAngle = math.pi - math.atan(
                (bot_y - top_y) / (bot_right_x - top_right_x)
            )
        # if lane is facing away from center
        else:
            self.rightAngle = math.atan(
                (bot_y - top_y) / (top_right_x - bot_right_x)
            )

    def check_valid_lanes(self):
        """
        Checks if each lane is within pi/2 and pi/8 (upper 3/4 of each
        quadrant). The values seem to work in most cases, subject to
        change.

        :return: nothing
        """
        if (math.pi / 8 <= self.leftAngle) \
                and (3 * math.pi / 8 > self.leftAngle) \
                and (7 * math.pi / 8 >= self.rightAngle) \
                and (5 * math.pi / 8 < self.rightAngle):
            self.leftCheck = True
            self.rightCheck = True
        else:
            return

    def check_lanes(self, bot_left_x, top_left_x, bot_right_x,
                    top_right_x, bot_y, top_y, img):
        """
        Displays if two lanes are valid in that their angles make sense.
        Does not check if they're actually in the correct place.
        Relies on calculate_angles and check_valid_lanes.

        :param bot_left_x: bottom left x coordinate
        :param top_left_x: top left x coordinate
        :param bot_right_x: bottom right x coordinate
        :param top_right_x: top right x coordinate
        :param bot_y: bottom of lanes
        :param top_y: top of lanes
        :param img: lane image
        :return: modified lane image with results
        """
        self.calculate_angles(bot_left_x, top_left_x, bot_right_x,
                              top_right_x, bot_y, top_y)
        self.check_valid_lanes()
        font = cv2.FONT_HERSHEY_SIMPLEX
        red = (255, 0, 0)
        green = (0, 255, 0)
        if self.leftCheck \
                and self.rightCheck \
                and top_left_x < top_right_x:       # lines don't cross
            cv2.putText(img, 'Lanes are valid', (100, 100), font, 1, green, 2)
        else:
            cv2.putText(img, 'Lanes are not valid', (100, 100), font, 1, red, 2)

