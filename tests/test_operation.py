import os
import torch
import torch_ipex
import torch_ccl
import tempfile
import unittest
import torch.distributed as c10d
from torch.testing._internal.common_utils import TestCase, run_tests

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class ProcessGroupOCCLTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = int(os.environ['PMI_RANK'])
        self.world_size = int(os.environ['PMI_SIZE'])
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_devs_per_proc = 1
        # self.num_devs_per_proc = torch.cuda.device_count()
        # if self.num_devs_per_proc < 2:
        #     print("skip test")
        #     raise unittest.SkipTest("OCCL test requires 2+ GPUs")

    def tearDown(self):
        pass

    # def test_empty_tensors(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     xs = [torch.FloatTensor([])]
    #     pg.broadcast(xs).wait()
    #     self.assertEqual(0, xs[0].numel())
    #
    #     pg.allreduce(xs).wait()
    #     self.assertEqual(0, xs[0].numel())
    #
    #     pg.reduce(xs).wait()
    #     self.assertEqual(0, xs[0].numel())
    #
    #     ys = [[torch.FloatTensor([]) for _ in range(self.world_size)]]
    #     pg.allgather(ys, xs).wait()
    #     for y in ys[0]:
    #         self.assertEqual(0, y.numel())
    #
    #     ys = [torch.FloatTensor([])]
    #     xs = [[torch.FloatTensor([]) for _ in range(self.world_size)]]
    #     pg.reduce_scatter(ys, xs).wait()
    #     self.assertEqual(0, ys[0].numel())

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root rank
        for dev_type in [cpu_device]:
            for rt in range(self.world_size):
                tensors = []
                tensors.append(torch.tensor([self.rank], device=dev_type))
                with torch.autograd.profiler.profile() as prof:
                    broadcast(tensors, rt, 0)
                print(prof)
                self.assertEqual(tensors[0], torch.tensor([rt]))


    # def test_allreduce_ops(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     def allreduce(tensors, op):
    #         opts = c10d.AllreduceOptions()
    #         opts.reduceOp = op
    #         work = pg.allreduce(tensors, opts)
    #         work.wait()
    #
    #     # Sum
    #     tensors = []
    #     for i in range(self.num_devs_per_proc):
    #         tensors.append(torch.tensor([i + 1]).cuda(i))
    #
    #     allreduce(tensors, c10d.ReduceOp.SUM)
    #
    #     for i in range(self.num_devs_per_proc):
    #         self.assertEqual(
    #             torch.tensor([float(self.num_devs_per_proc * (self.num_devs_per_proc + 1) / 2)]),
    #             tensors[i])
    #
    #     # Product
    #     tensors = []
    #     for i in range(self.num_devs_per_proc):
    #         tensors.append(torch.tensor([i + 1]).cuda(i))
    #
    #     allreduce(tensors, c10d.ReduceOp.PRODUCT)
    #
    #     for i in range(self.num_devs_per_proc):
    #         self.assertEqual(
    #             torch.tensor([float(math.factorial(self.num_devs_per_proc))]),
    #             tensors[i])
    #
    #     # Min
    #     tensors = []
    #     for i in range(self.num_devs_per_proc):
    #         tensors.append(torch.tensor([i + 1]).cuda(i))
    #
    #     allreduce(tensors, c10d.ReduceOp.MIN)
    #
    #     for i in range(self.num_devs_per_proc):
    #         self.assertEqual(torch.tensor([1.0]), tensors[i])
    #
    #     # Max
    #     tensors = []
    #     for i in range(self.num_devs_per_proc):
    #         tensors.append(torch.tensor([i + 1]).cuda(i))
    #
    #     allreduce(tensors, c10d.ReduceOp.MAX)
    #
    #     for i in range(self.num_devs_per_proc):
    #         self.assertEqual(torch.tensor([self.num_devs_per_proc]), tensors[i])

    # def test_reduce_ops(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     def reduce(xs, rootRank, rootTensor):
    #         opts = c10d.ReduceOptions()
    #         opts.rootRank = rootRank
    #         opts.rootTensor = rootTensor
    #         work = pg.reduce(xs, opts)
    #         work.wait()
    #
    #     # for every root tensor
    #     for rt in range(self.num_devs_per_proc):
    #         tensors = []
    #         for i in range(self.num_devs_per_proc):
    #             tensors.append(torch.tensor([i + 1]).cuda(i))
    #
    #         reduce(tensors, self.rank, rt)
    #
    #         self.assertEqual(
    #             torch.tensor([float(self.num_devs_per_proc * (self.num_devs_per_proc + 1) / 2)]),
    #             tensors[rt])

    # def test_allgather_ops(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     def allgather(output_ts, input_ts):
    #         work = pg.allgather(output_ts, input_ts)
    #         work.wait()
    #
    #     tensors = []
    #     output_ts = [[] for _ in range(self.num_devs_per_proc)]
    #
    #     for idx, ls in enumerate(output_ts):
    #         for _ in range(self.world_size * self.num_devs_per_proc):
    #             ls.append(torch.tensor([0]).cuda(idx))
    #
    #     for i in range(self.num_devs_per_proc):
    #         tensors.append(torch.tensor([i]).cuda(i))
    #
    #     allgather(output_ts, tensors)
    #
    #     # Verification
    #     for device_ts in output_ts:
    #         for s_idx, t in enumerate(device_ts):
    #             self.assertEqual(torch.tensor([s_idx]), t)

    # def test_reduce_scatter_ops(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     def reduce_scatter(outputs, input_lists, op):
    #         opts = c10d.ReduceScatterOptions()
    #         opts.reduceOp = op
    #         work = pg.reduce_scatter(outputs, input_lists, opts)
    #         work.wait()
    #
    #     virtual_rank = self.rank * self.world_size
    #     virtual_world_size = self.num_devs_per_proc * self.world_size
    #
    #     output = [
    #         torch.tensor([0])
    #         for i in range(self.num_devs_per_proc)
    #     ]
    #
    #     #           0                   1                   2
    #     #   0   [0..11]             [1..12]
    #     #   1   [3..14]
    #     #   2
    #     #   3
    #
    #     # Sum
    #     tensor_lists = [
    #         [
    #             torch.tensor([self.rank * self.num_devs_per_proc + i + j])
    #             for j in range(virtual_world_size)
    #         ]
    #         for i in range(self.num_devs_per_proc)
    #     ]
    #
    #     reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)
    #
    #     for i in range(self.num_devs_per_proc):
    #         expected = torch.tensor([
    #             float(self.num_devs_per_proc * (self.num_devs_per_proc - 1) / 2) +
    #             (virtual_rank + i) * virtual_world_size
    #         ])
    #         self.assertEqual(expected, output[i])
    #
    #     # Min
    #     reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)
    #
    #     for i in range(self.num_devs_per_proc):
    #         expected = torch.tensor([self.rank * self.world_size + i])
    #         self.assertEqual(expected, output[i])
    #
    #     # Max
    #     reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)
    #
    #     for i in range(self.num_devs_per_proc):
    #         expected = torch.tensor(
    #             [self.rank * self.world_size + i + virtual_world_size - 1]
    #         )
    #         self.assertEqual(expected, output[i])
    #
    #     # Product
    #     tensor_lists = [
    #         [
    #             torch.tensor([
    #                 (self.rank * self.num_devs_per_proc + i + j) % virtual_world_size + 1
    #             ])
    #             for j in range(virtual_world_size)
    #         ]
    #         for i in range(self.num_devs_per_proc)
    #     ]
    #
    #     reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)
    #
    #     for i in range(self.num_devs_per_proc):
    #         expected = torch.tensor([float(math.factorial(virtual_world_size))])
    #         self.assertEqual(expected, output[i])

    # def test_barrier(self):
    #     store = c10d.FileStore(self.file.name, self.world_size)
    #     pg = c10d.ProcessGroupOCCL(store, self.rank, self.world_size)
    #
    #     def allreduce(tensors):
    #         opts = c10d.AllreduceOptions()
    #         work = pg.allreduce(tensors, opts)
    #         return work
    #
    #     # Making the collective to operate on
    #     # 1, 2, 3, 4, .... self.num_devs_per_proc GPUs
    #     tensors_list = [[] for _ in range(2, self.num_devs_per_proc + 1)]
    #     for i in range(2, self.num_devs_per_proc + 1):
    #         for j in range(i):
    #             tensors_list[i - 2].append(torch.tensor([j + 1]).cuda(j))
    #
    #     works = []
    #     for tensors in tensors_list:
    #         work = allreduce(tensors)
    #         works.append(work)
    #
    #     # Barrier will ensure that all previous work is completed
    #     pg.barrier().wait()
    #
    #     for i in range(2, self.num_devs_per_proc + 1):
    #         for j in range(i):
    #             self.assertEqual(
    #                 torch.tensor([float(i * (i + 1) / 2)]),
    #                 tensors_list[i - 2][j])


if __name__ == '__main__':
    # assert not torch.cuda._initialized, "test_distributed must not have initialized CUDA context on main process"

    run_tests()