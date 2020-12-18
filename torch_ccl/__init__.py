import sys
import warnings
from .version import __version__, build_type, git_version

from . import _C as occl_lib

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

