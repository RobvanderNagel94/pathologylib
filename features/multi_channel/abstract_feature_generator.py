from abc import ABC


class AbstractFeatureGenerator(ABC):

    def __init__(self, domain, electrodes, agg_mode):
        self.domain = domain
        self.electrodes = electrodes
        self.agg_mode = agg_mode
        self.times_list = []
