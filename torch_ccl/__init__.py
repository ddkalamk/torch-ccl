import sys
import warnings

from .lib import libtorch_ccl as occl_lib

# from .lib import libtorch_ccl_dpcpp as occl_lib



from .occl import (
    all_reduce,
    all_gather,
    broadcast,
    init_rank,
    reduce,
    reduce_scatter,
    unique_id)

__all__ = []
__all__ += [name for name in dir(occl_lib)
            if name[0] != '_' and
            not name.endswith('Base')]


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if not tensor.is_contiguous():
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True

