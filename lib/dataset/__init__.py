from ._360cc import _360CC
from ._own import _OWN

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()