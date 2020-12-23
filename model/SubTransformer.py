import torch as t
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm


class SubTransformer(Module):
    def __init__(self, d_model, opt):
        super().__init__()
        self.opt = opt
        nhead = d_model
        dropout = opt.drop_out
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear = Linear(d_model * 3, opt.conv_channel)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model * 3)

    def forward(self, query, src, batch):
        src2 = []
        batch_unique = set(batch.detach().tolist())
        for graph in range(len(batch_unique)):
            graph_idx = t.nonzero(batch.detach() == graph, as_tuple=False).squeeze().detach()
            out = self.self_attn(query[graph_idx], src[graph_idx], src[graph_idx])[0]
            src2.append(out)
        src2 = t.cat(src2, dim=0)
        # src2 = self.norm1(src2)
        
        src = t.cat([src2, query, src], dim=-1)
        src = self.norm2(src)
        src = t.nn.functional.leaky_relu(self.dropout1(self.linear(src)))
        return src
