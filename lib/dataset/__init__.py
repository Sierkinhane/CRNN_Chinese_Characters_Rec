from ._360cc import _360CC

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    else:
        raise NotImplemented()