from torch import nn
from PatchTST_backbone import PatchTST_backbone


class Model(nn.Module):
    def __init__(self, configs, act: str = "gelu", res_attention: bool = True, head_type='flatten'):
        super().__init__()

        # load parameters
        c_in = configs.enc_in  # 4
        context_window = configs.seq_len  # 输入序列长度
        target_window = configs.pred_len  # 预测序列长度

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model  # 每个patch的编码维度
        d_ff = configs.d_ff
        dropout = configs.dropout
        # fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        patch_len = configs.patch_len  # patch的长度
        stride = configs.stride  # 相邻两个patch之间的长度
        padding_patch = configs.padding_patch  # 填充的方式“end”

        self.args = configs

        self.liner1 = nn.Linear(4 * configs.seq_len, 2 * configs.seq_len)
        self.liner2 = nn.Linear(2 * configs.seq_len, configs.seq_len)
        self.liner3 = nn.Linear(configs.seq_len, 2)

        self.model = PatchTST_backbone(configs, c_in=c_in, context_window=context_window, target_window=target_window,
                                       patch_len=patch_len, stride=stride,
                                       n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_ff=d_ff,
                                       dropout=dropout, act=act,
                                       res_attention=res_attention,
                                       head_dropout=head_dropout,
                                       padding_patch=padding_patch,
                                       head_type=head_type)

    def forward(self, x):  # x: [Batch, Input length, Channel]

        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        # print(x.size())
        x = self.model(x)  # bs x nvars x target_window]

        return x
