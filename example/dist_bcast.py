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
    t = torch.FloatTensor([rank])
    print("before broadcast rank {} size {} result {}".format(rank, size, t.cpu()))
    torch.distributed.broadcast(t, src=0)
    print("after broadcast rank {} size {} result {}".format(rank, size, t.cpu()))



def init_process(rank, size, fn, backend='occl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":

    # print(os.environ)
    rank = int(os.environ['PMI_RANK'])
    size = int(os.environ['PMI_SIZE'])
    print("rank {} size {}".format(rank, size))
    init_process(rank, size, run)
    # init_process()
    # size = 2
    # processes = []
    # for rank in range(size):
    #     p = Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()