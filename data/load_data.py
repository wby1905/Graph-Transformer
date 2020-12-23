import os

import torch as t
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm

from config import opt
from .mapping import qw_score, pre_processing, construct_node_tree


# from DGCNN
class Indegree(object):
    r"""Adds the globally normalized node degree to the node features.
    Args:
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        col, x = data.edge_index[1], data.x
        deg = degree(col, data.num_nodes)

        if self.norm:
            deg = deg / (deg.max() if self.max is None else self.max)

        deg = deg.view(-1, 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = t.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__, self.norm, self.max)


class TUD(Dataset):

    def __init__(self):
        rt = opt.save_processed_dir + '/' + opt.dataset + '/' f'{opt.K}_{opt.m}' + '/'

        if not os.path.exists(rt):
            root = opt.dataset_dir
            dataset = TUDataset(root, name=opt.dataset, use_node_attr=True, pre_transform=Indegree())
            opt.input_features = dataset.num_node_features
            self.x_w_trees = []  # 每一张图的w个主节点的树
            self.graph_list = []

            # 处理每一张图
            print('preprocessing..\n')
            wrong = 0
            for idx, data in enumerate(tqdm(dataset)):
                # 预生成所有待测节点的m叉k层树
                trees = {i: [] for i in range(data.num_nodes)}
                score = qw_score(data)
                x = data.x.clone()
                data.x = t.arange(0, data.num_nodes)[:, None]
                indices = reversed(score.indices)
                trees = pre_processing(data, opt.m, score, trees)
                graph_trees = []
                for node in indices:
                    tree = construct_node_tree(data, node.item(), trees, opt)
                    graph_trees.append(tree[None, :])
                graph_trees = t.cat(graph_trees, dim=0)
                data.x_w_trees = graph_trees
                w = opt.W
                if opt.W == 'all' or opt.W > x.shape[0]:  # 大于无意义
                    w = data.x.shape[0]
                tmp = t.zeros(data.x_w_trees.shape)
                ori_idx = t.tensor(indices[:w]).clone().detach()
                tmp[ori_idx] = data.x_w_trees[:w].clone().detach()
                if x.shape[0] != x[tmp.long()].shape[0]:
                    print(idx, x.shape, tmp.shape)
                    wrong += 1
                    continue
                data.x_w_trees = tmp.clone().detach()
                data.x = x
                save_graph(data, idx - wrong)
            del dataset

        opt.input_features = self.__getitem__(0).x_w_trees.shape[2]
        labels = {
            'ENZYMES': 6,
            'IMDB-MULTI': 3
        }
        opt.classes = labels.get(opt.dataset, 2)
        self.count = len([lists for lists in os.listdir(rt) if os.path.isfile(os.path.join(rt, lists))])
        print(self.count, opt.classes)
        print('Load Dataset Done!')


    def __getitem__(self, index: int):
        rt = opt.save_processed_dir + '/' + opt.dataset + '/' f'{opt.K}_{opt.m}' + '/'
        path = rt + '/' + f'{index}.pth'
        graph = t.load(path)
        graph.x_w_trees = graph.x[graph.x_w_trees.squeeze().long()]
        return graph

    def __len__(self) -> int:
        return self.count


def save_graph(graph, idx):
    rt = opt.save_processed_dir + '/' + opt.dataset + '/' f'{opt.K}_{opt.m}' + '/'
    if not os.path.exists(opt.save_processed_dir + '/' + opt.dataset):
        os.makedirs(opt.save_processed_dir + '/' + opt.dataset)
    if not os.path.exists(rt):
        os.mkdir(rt)
    t.save(graph, rt + f'{idx}.pth')
