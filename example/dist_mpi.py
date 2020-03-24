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
    t = torch.FloatTensor([1]).to("dpcpp")
    torch.distributed.all_reduce(t)

    print("john 44 rank {} size {}".format(rank, size))
    print("result ", t.cpu())
    # fc = torch.nn.Linear(3, 3)
    # if rank == 0:
    #     fc.weight = torch.nn.Parameter(torch.ones([3, 3]))
    #     fc.bias = torch.nn.Parameter(torch.ones([3, 3]))
    # else:
    #     fc.weight = torch.nn.Parameter(torch.zeros([3, 3]))
    #     fc.bias = torch.nn.Parameter(torch.zeros([3, 3]))
    # fc.to("dpcpp")
    # dist_fc = torch.nn.parallel.DistributedDataParallel(fc)
    #
    # print("john 33 rank {} size {}".format(rank, size))
    # print(type(rank))
    #
    # if rank == 1:
    #     print("john 22 rank {} size {}".format(rank, size))
    #     print(fc.state_dict().values())
    # if rank == 1:
    #     print("john 22 rank {} size {}".format(rank, size))
    #     print(dist_fc.state_dict().values())
    #
    # print("john 44 rank {} size {}".format(rank, size))



def init_process(rank, size, fn, backend='occl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    print("john  rank {} size {}".format(rank, size))


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