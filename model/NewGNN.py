import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import remove_self_loops

from model.SubGraphConv import SubGraphConv
from model.SubGraphPool import SubGraphAvgPool
from model.SubTransformer import SubTransformer


class GNN(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.sub_conv = nn.ModuleList([SubGraphConv(opt, opt.input_features, opt.conv_channel)])
        self.sub_conv.extend([
            SubGraphConv(opt, opt.conv_channel, opt.conv_channel) for _ in range(opt.conv_num - 2)
        ])
        self.sub_conv.append(SubGraphAvgPool(opt))

        in_gcn = nn.ModuleList([GCNConv(opt.input_features, opt.conv_channel)])
        in_gcn.extend([GCNConv(opt.conv_channel, opt.conv_channel) for _ in range(opt.gcn_num - 1)])
        self.graph_conv = nn.ModuleList([in_gcn])
        self.graph_conv.extend([
            nn.ModuleList(
                [
                    GCNConv(opt.conv_channel, opt.conv_channel) for _ in range(opt.gcn_num)
                ]
            ) for _ in range(opt.K)
        ])

        self.color_gcn = nn.ModuleList(
            [
                GCNConv(opt.conv_channel, 1) for _ in range(opt.gcn_num + 1)
            ]
        )

        self.sub_trans = nn.ModuleList(
            [
                SubTransformer(opt.input_features, opt)
            ]
            +
            [
                SubTransformer(opt.conv_channel, opt)
                for _ in range(opt.conv_num)
            ]
        )

        self.conv1d = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1d(1, opt.conv1d_channel, opt.K * opt.conv_channel + 1 + opt.input_features,
                           opt.K * opt.conv_channel + 1 + opt.input_features),
                    nn.LeakyReLU(),
                    nn.AvgPool1d(2, 2),
                    nn.LeakyReLU(),
                    Conv1d(opt.conv1d_channel, opt.conv1d_channel * 2, 5, 1),
                    nn.LeakyReLU()
                )
            ]
        )
        self.conv1d.extend([
            nn.Sequential(
                Conv1d(1, opt.conv1d_channel, ((opt.K + 1) * opt.conv_channel) * i + opt.K * opt.conv_channel + 1 * (i + 1) + opt.input_features,
                       ((opt.K + 1) * opt.conv_channel) * i + opt.K * opt.conv_channel + 1 * (i + 1) + opt.input_features),
                nn.LeakyReLU(),
                nn.AvgPool1d(2, 2),
                nn.LeakyReLU(),
                Conv1d(opt.conv1d_channel, opt.conv1d_channel * 2, 5, 1),
                nn.LeakyReLU(),

            ) for i in range(1, opt.gcn_num + 1)
        ]
        )

        self.classifier = nn.ModuleList(
            [
                nn.Linear(opt.conv1d_channel * (opt.top_k - 8) * (opt.gcn_num + 1), opt.fc_channel),
                nn.Dropout(opt.drop_out)
            ]
        )

        for _ in range(opt.fc_layers_num - 2):
            self.classifier.extend([
                nn.Linear(opt.fc_channel, opt.fc_channel, bias=True),
                nn.Dropout(opt.drop_out * 0.5)
            ])

        self.classifier.append(nn.Linear(opt.fc_channel, opt.classes))
        
        self.loss = LabelSmoothingCrossEntropy(opt.eps)

    def forward(self, data):
        batch_trees = data.x_w_trees
        batch_x = data.x
        batch_edge_index = data.edge_index
        batch_edge_index, _ = remove_self_loops(batch_edge_index)
        batch = data.batch  # 大小为全部batch中的点，数值表示属于哪一张图
        x = [[] for _ in range(self.opt.gcn_num + 1)]
        for now in range(self.opt.K):

            # # 图卷积
            g_x = batch_x
            for i, gcn in enumerate(self.graph_conv[now]):
                x[i].append(g_x)
                g_x = t.tanh(gcn(g_x, batch_edge_index))  # all_nodes x feature_out
            # 将输出记录
            x[-1].append(g_x)

            # 计算attention
            batch_x = batch_x.unsqueeze(1)
            batch_tree_root = batch_trees[:, 0, :].unsqueeze(1)  # 取根节点的特征 all_nodes x features
            batch_x = self.sub_trans[now](batch_tree_root, batch_x, batch).squeeze()

            # 子树卷积
            if now < self.opt.conv_num:
                batch_trees = self.sub_conv[now](batch_trees)  # all_nodes x new_tree_nodes x feature_out

        g_x = batch_x
        for i, gcn in enumerate(self.graph_conv[-1]):
            x[i].append(g_x)
            g_x = t.tanh(gcn(g_x, batch_edge_index))  # all_nodes x feature_out
        # 将输出记录
        x[-1].append(g_x)

        for i, gcn in enumerate(self.color_gcn):
            color = t.tanh(gcn(x[i][-1], batch_edge_index))  # 最后输出点的颜色  all_nodes (x 1)
            x[i].append(color)

        x = [t.cat(i, dim=-1) for i in x]
        concat = [[] for _ in range(self.opt.gcn_num + 1)]
        for i in range(self.opt.gcn_num + 1):
            for j in range(i):
                concat[i].append(x[j])
            concat[i].append(x[i])
            concat[i] = t.cat(concat[i], dim=-1)
            concat[i] = global_sort_pool(concat[i], batch, k=self.opt.top_k)

        # 剩余结构与 DGCNN 相同
        for i, layer in enumerate(self.conv1d):
            concat[i] = layer(concat[i].unsqueeze(1))
            concat[i] = concat[i].view(concat[i].shape[0], -1)  # 展开 喂入MLP

        concat = t.cat(concat, dim=-1)
        # print(concat.shape)
        for layer in self.classifier:
            concat = F.leaky_relu(layer(concat))
        return concat, self.loss(concat, data.y), data.y


class LabelSmoothingCrossEntropy(t.nn.Module):
    def __init__(self, eps=0, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds, targets):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(F.log_softmax(preds, dim=-1))
        nll = F.nll_loss(log_preds, targets, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.eps)


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, eps):
    return eps * x + (1 - eps) * y
