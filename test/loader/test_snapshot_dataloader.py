import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import SnapshotLoader


@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_snapshot_dataloader(num_workers):
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4], [1, 0, 2, 1, 3, 2, 5]], dtype=torch.long
    )
    time = torch.tensor([0, 1, 1, 1, 2, 2, 3], dtype=torch.long)
    data = Data(edge_index=edge_index, time=time, num_nodes=6)

    loader = SnapshotLoader(data, horizon=1, num_workers=num_workers)
    assert len(loader) == 4

    for i, snapshot in enumerate(loader):
        assert snapshot.edge_index.tolist() == edge_index[:, time == i].tolist()
        assert snapshot.time.tolist() == time[time == i].tolist()
        assert snapshot.num_nodes == 6

    loader = SnapshotLoader(data, horizon=2, num_workers=num_workers)
    assert len(loader) == 2

    for i, snapshot in enumerate(loader):
        assert (
            snapshot.edge_index.tolist()
            == edge_index[:, (time == i * 2) | (time == (i * 2) + 1)].tolist()
        )
        assert (
            snapshot.time.tolist()
            == time[(time == i * 2) | (time == (i * 2) + 1)].tolist()
        )
        assert snapshot.num_nodes == 6
