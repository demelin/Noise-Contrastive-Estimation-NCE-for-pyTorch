import os


class NCEOptions(object):
    """ Default options for the noise contrastive estimation loss criterion. Modify as needed. """
    def __init__(self):
        self.num_sampled = 25
        self.remove_accidental_hits = True
        self.subtract_log_q = True
        self.unique = True
        self.array_path = os.path.join(os.curdir, 'sampling_array.p')
