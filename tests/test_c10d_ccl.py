import torch
import torch_ccl
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase
import torch.distributed as c10d

import math
from functools import reduce
import operator

cpu_device = torch.device("cpu")

def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = (1 << i)
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    return tests


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2 * world_size]),
        ),
    ]


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

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([
                    (i * self.world_size) + (i % self.world_size)
                ]),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_broadcast_stress(self):
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(output, tensor)

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            work = pg.allreduce(tensors, opts)
            work.wait()
            for tensor in tensors:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), x)

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())


if __name__ == '__main__':

    run_tests()
