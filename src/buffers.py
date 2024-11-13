class RandomBuffer:
    def __init__(self, n_samples: int):
        """Stores random samples for replay."""
        pass


class RingBuffer:
    def __init__(self, n_samples: int):
        """Stores the most recent samples for replay."""
        pass


class MeanOfFeatureBuffer:
    def __init__(self, n_samples: int):
        """Stores samples closest to the class centroids in the feature space."""
        pass
