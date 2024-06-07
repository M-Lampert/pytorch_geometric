import torch

from torch_geometric.data import Data


class IterableSnapshotDataset(torch.utils.data.IterableDataset):
    r"""An iterable dataset that yields snapshots of events that
    occured in the period `[t, t+horizon-1]`

    Args:
        data (Data): The :obj:`~torch_geometric.data.Data`
            from which to load the data.
        horizon (int): The time horizon for each snapshot.
    """

    def __init__(self, data, horizon):
        self.data = data.sort_by_time()
        self.horizon = horizon
        self.num_workers = 1
        self.worker_id = 0
        self.time_min = int(self.data.time.min().floor())
        self.time_max = int(self.data.time.max().ceil())

    def __len__(self):
        return (self.time_max - self.time_min) // self.horizon + 1

    def __getitem__(self, index):
        return self.data.snapshot(
            start_time=self.time_min + (index * self.horizon),
            end_time=self.time_min + ((index + 1) * self.horizon) - 1,
        )

    def __iter__(self):
        for i in range(
            self.worker_id,
            len(self),
            self.num_workers,
        ):
            yield self[i]


class SnapshotLoader(torch.utils.data.DataLoader):
    r"""A data loader which loads a snaphshot that holds events that
    occured in the period `[t, t+horizon-1]`

    Args:
        data (Data): The :obj:`~torch_geometric.data.Data`
            from which to load the data.
        horizon (int): The time horizon for each snapshot.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        data: Data,
        horizon: int,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)
        kwargs.pop("shuffle", None)

        iterable_dataset = IterableSnapshotDataset(data=data, horizon=horizon)

        def _worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            dataset.num_workers = worker_info.num_workers
            dataset.worker_id = worker_id

        super().__init__(
            iterable_dataset,
            batch_size=None,
            shuffle=False,
            worker_init_fn=_worker_init_fn,
            **kwargs,
        )
