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

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root rank
        for dev_type in [cpu_device]:
            for rt in range(self.world_size):
                tensor = torch.tensor([self.rank], device=dev_type)
                broadcast([tensor], rt, 0)
                self.assertEqual(tensor, torch.tensor([rt]))



if __name__ == '__main__':

    run_tests()
