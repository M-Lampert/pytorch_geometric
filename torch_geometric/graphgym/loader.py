from typing import Callable

import torch

import torch_geometric.graphgym.register as register
import torch_geometric.transforms as T
from torch_geometric.data.lightning_datamodule import (
    LightningLinkData,
    LightningNodeData,
)
from torch_geometric.datasets import (
    PPI,
    Amazon,
    Coauthor,
    KarateClub,
    MNISTSuperpixels,
    Planetoid,
    QM7b,
    TUDataset,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)

index2mask = index_to_mask  # TODO Backward compatibility


def planetoid_dataset(name: str) -> Callable:
    return lambda root: Planetoid(root, name)


register.register_dataset('Cora', planetoid_dataset('Cora'))
register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
register.register_dataset('PubMed', planetoid_dataset('PubMed'))
register.register_dataset('PPI', PPI)


def load_pyg(arg_dict):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        arg_dict (dict): Dictionary with name, dataset directory and optionally transforms

    Returns: PyG dataset object

    """
    name = arg_dict["name"]
    arg_dict["root"] = '{}/{}'.format(arg_dict["root"], arg_dict["name"])
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(**arg_dict)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            arg_dict["name"] = 'IMDB-MULTI'
            arg_dict["transform"] = T.Constant() if arg_dict["transform"] is None else T.Compose([arg_dict["transform"], T.Constant()])
            dataset = TUDataset(**arg_dict)
        else:
            arg_dict["name"] = arg_dict["name"][3:]
            dataset = TUDataset(**arg_dict)
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            arg_dict["name"] = "CS"
            dataset = Coauthor(**arg_dict)
        else:
            arg_dict["name"] = 'Physics'
            dataset = Coauthor(**arg_dict)
    elif 'Amazon' in name:
        if 'Computers' in name:
            arg_dict["name"] = 'Computers'
            dataset = Amazon(**arg_dict)
        else:
            arg_dict["name"] = 'Photo'
            dataset = Amazon(**arg_dict)
    elif name == 'MNIST':
        del arg_dict["name"]
        dataset = MNISTSuperpixels(**arg_dict)
    elif name == 'PPI':
        del arg_dict["name"]
        dataset = PPI(**arg_dict)
    elif name == 'QM7b':
        del arg_dict["name"]
        dataset = QM7b(**arg_dict)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


def set_dataset_attr(dataset, name, value, size):
    dataset._data_list = None
    dataset.data[name] = value
    if dataset.slices is not None:
        dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


def load_ogb(arg_dict):
    r"""

    Load OGB dataset objects.


    Args:
        arg_dict (dict): Dictionary with name, dataset directory and optionally transforms

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if arg_dict["name"][:4] == 'ogbn':
        dataset = PygNodePropPredDataset(**arg_dict)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        edge_index = to_undirected(dataset.data.edge_index)
        set_dataset_attr(dataset, 'edge_index', edge_index,
                         edge_index.shape[1])

    elif arg_dict["name"][:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(**arg_dict)
        splits = dataset.get_idx_split()
        split_names = ['train_index', 'val_index', 'test_index']
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif arg_dict["name"][:4] == "ogbl":
        dataset = PygLinkPropPredDataset(**arg_dict)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset


def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    arg_dict = {
        "name": cfg.dataset.name,
        "root": cfg.dataset.dir
    }
    if len(cfg.dataset.transform) > 0:
        transforms = []
        for curr_transform in cfg.dataset.transform:
            transform_cls = register.transform_dict.get(curr_transform)
            if transform_cls is not None:
                transforms.append(transform_cls())
            else:
                raise ValueError(
                    f'Transform function list contains unknown transform: {cfg.dataset.transform}')
        if len(transforms) > 1:
            arg_dict["transform"] = T.Compose(transforms)
        else:
            arg_dict["transform"] = transforms[0]
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, **arg_dict)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(arg_dict)
    # Load from OGB formatted data
    elif format == 'OGB':
        arg_dict["name"] = arg_dict["name"].replace('_', '-')
        dataset = load_ogb(arg_dict)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset


def set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """

    # get dim_in and dim_out
    try:
        cfg.share.dim_in = dataset.data.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
    try:
        if cfg.dataset.task_type == 'classification':
            cfg.share.dim_out = torch.unique(dataset.data.y).shape[0]
        else:
            cfg.share.dim_out = dataset.data.y.shape[1]
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset.data.keys:
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset.data.keys:
        if 'test' in key:
            cfg.share.num_splits += 1
            break


def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def create_loader(dataset):
    """
    Create a LightningDataset object

    Args:
        dataset (torch_geometric.data.dataset.Dataset): The dataset that should be used to create a loader.

    Returns: PyG LightningDataset object

    """
    # Provide backward compatibility for old config files
    if cfg.train.sampler == "full_batch":
        cfg.train.sampler = "full"

    data = dataset[0]

    if cfg.train.sampler not in ["full", "neighbor", "link_neighbor"]:
        raise NotImplementedError()

    if cfg.dataset.task in ['link_pred', 'edge']:

        train_edges = data.get("train_edge_index", None)
        train_labels = data.get("train_edge_label", None)
        val_edges = data.get("val_edge_index", None)
        val_labels = data.get("val_edge_label", None)
        test_edges = data.get("test_edge_index", None)
        test_labels = data.get("test_edge_label", None)

        del data["train_edge_index"]
        del data["train_edge_label"]
        del data["val_edge_index"]
        del data["val_edge_label"]
        del data["test_edge_index"]
        del data["test_edge_label"]

        return LightningLinkData(
            data=data, loader=cfg.train.sampler, input_train_edges=train_edges,
            input_train_labels=train_labels, input_val_edges=val_edges,
            input_val_labels=val_labels, input_test_edges=test_edges,
            input_test_labels=test_labels, batch_size=cfg.train.batch_size,
            num_workers=cfg.num_workers,
            num_neighbors=cfg.train.neighbor_sizes)
    else:
        return LightningNodeData(data=data, loader=cfg.train.sampler,
                                 batch_size=cfg.train.batch_size,
                                 num_workers=cfg.num_workers,
                                 num_neighbors=cfg.train.neighbor_sizes)
