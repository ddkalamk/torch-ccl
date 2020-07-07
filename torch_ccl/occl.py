import sys
import warnings


SUM = 0  # ncclRedOp_t


def unique_id():
    return torch._C._nccl_unique_id()


def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)


def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    if outputs is None:
        outputs = inputs
    torch._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


def reduce(inputs, outputs=None, root=0, op=SUM, streams=None, comms=None):
    if outputs is None:
        outputs = inputs
    torch._C._nccl_reduce(inputs, outputs, root, op, streams, comms)


def broadcast(inputs, root=0, streams=None, comms=None):
    torch._C._nccl_broadcast(inputs, root, streams, comms)


def all_gather(inputs, outputs, streams=None, comms=None):
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)


def reduce_scatter(inputs, outputs, op=SUM, streams=None, comms=None):
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)
