import torch
import torch_ccl
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase
import torch.distributed as c10d

cpu_device = torch.device("cpu")


class ProcessGroupCCLTest(MultiProcessTestCase):

    def setUp(self):
        super(ProcessGroupCCLTest, self).setUp()
        self._fork_processes()

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "unexpected rank"):

            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)


    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            broadcast([x], i, 0)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([i]), x)

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                broadcast(xs, i, j)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), xs[0])
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), xs[1])

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())




if __name__ == '__main__':

    run_tests()
