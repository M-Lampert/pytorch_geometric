import torch

from torch_geometric.data import Data
from torch_geometric.utils import cumsum


class IterableSnapshotDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, horizon, max_batch_size):
        self.data = data.sort_by_time()
        self.horizon = horizon
        self.max_batch_size = max_batch_size
        # Compute the pointers to the snapshots with maximum batch size
        _, n_stamps_in_window = torch.unique(
            torch.floor(self.data.time / self.horizon), return_counts=True
        )
        steps_per_window = torch.ceil(n_stamps_in_window / self.max_batch_size).int()
        self.ptrs = torch.repeat_interleave(
            cumsum(n_stamps_in_window)[:-1], steps_per_window
        )
        correction = torch.arange(self.ptrs.shape[0], device=self.ptrs.device)
        correction -= torch.repeat_interleave(
            cumsum(steps_per_window)[:-1], steps_per_window
        )
        self.ptrs += correction * self.max_batch_size
        self.steps = torch.arange(self.ptrs.shape[0], device=self.ptrs.device)
        self.ptrs = torch.cat(
            [
                self.ptrs,
                torch.tensor([self.data.time.shape[0]], device=self.ptrs.device),
            ]
        )

    def __iter__(self):
        for i in self.steps:
            yield self.data.edge_subgraph(
                torch.arange(
                    self.ptrs[i], self.ptrs[i + 1], device=self.data.edge_index.device
                )
            )


class SnapshotLoader(torch.utils.data.DataLoader):
    r"""A data loader which loads a snaphshot that holds events that
    occured in the period `[t, t+horizon-1]`

    Args:
        data (Data): The :obj:`~torch_geometric.data.Data`
            from which to load the data.
        horizon (int): The time horizon for each snapshot.
        max_batch_size (int): The maximum number of events in a batch.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        data: Data,
        horizon: int,
        max_batch_size: int = 1,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)
        kwargs.pop("shuffle", None)

        iterable_dataset = IterableSnapshotDataset(
            data=data, horizon=horizon, max_batch_size=max_batch_size
        )

        def worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            num_workers = worker_info.num_workers
            dataset.steps = dataset.steps[worker_id::num_workers]

        super().__init__(
            iterable_dataset,
            batch_size=None,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            **kwargs,
        )
