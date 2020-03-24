"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch_ipex
import torch.distributed as dist
import occl
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass
    # fc = torch.nn.Linear(3, 3)
    #
    # if rank == 0:
    #     fc.weight = torch.nn.Parameter(torch.ones([3, 3]))
    #     fc.bias = torch.nn.Parameter(torch.ones([3, 3]))
    # else:
    #     fc.weight = torch.nn.Parameter(torch.zeros([3, 3]))
    #     fc.bias = torch.nn.Parameter(torch.zeros([3, 3]))
    #
    # if rank == 1:
    #     print(fc.state_dict().values())
    # dist_fc = torch.nn.parallel.DistributedDataParallel(fc)
    # if rank == 1:
    #     print(dist_fc.state_dict().values())



def init_process(rank, size, fn, backend='occl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 1
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()