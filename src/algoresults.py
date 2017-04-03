class AlgoResult:
    def __init__(self, algorithm):
        self.left_alg = algorithm
        self.right_alg = algorithm
        self.left_thresh = None
        self.right_thresh = None
        self.left_fit = None
        self.right_fit = None
        self.left_conf = 0
        self.right_conf = 0
        self.conf = None
        self.left_warp_Minv = None
        self.right_warp_Minv = None