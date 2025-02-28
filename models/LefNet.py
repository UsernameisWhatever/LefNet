import copy
import numpy as np
import torch
import torch.nn as nn
import pickle


class ModelSingle(nn.Module):
    def __init__(self, args, input_dim=333, output_dim=83, data_order=2, k=2):
        super(ModelSingle, self).__init__()
        # (batch_size, seq_len, input_dim, k, out_dim)
        self.args = args
        self.batch_size = self.args.batch_size
        self.seq_len = self.args.seq_len
        self.input_dim = input_dim
        self.k = k
        self.output_dim = output_dim
        self.scale = self.args.scale
        self.data_order = data_order
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # prune related parameters
        self.threshold = self.args.prune_threshold
        self.feature_pair_list = []

        if self.data_order == 2:
            self.taylor_a_0_1 = nn.Parameter(self.scale * torch.randn(1, self.seq_len,
                                                                      self.input_dim, self.k, self.output_dim)
                                             )
            self.mask_paras = torch.ones(1, self.seq_len, self.input_dim, self.k, self.output_dim
                                         ).to(device=self.device).requires_grad_(False)
            self.taylor_a_0 = nn.Parameter(torch.randn(1, self.seq_len, self.input_dim,
                                                       self.output_dim))
            self.mask_a_0 = torch.ones(1, self.seq_len, self.input_dim, self.output_dim
                                       ).to(device=self.device).requires_grad_(False)
            self.final_bias = torch.zeros(1, self.seq_len, self.output_dim).to(self.device).requires_grad_(False)
        elif self.data_order == 1:
            self.taylor_a_0_1 = nn.Parameter(self.scale * torch.randn(1, self.input_dim, self.k, self.output_dim)
                                             )
            self.mask_paras = torch.ones(1, self.input_dim, self.k, self.output_dim
                                         ).to(device=self.device).requires_grad_(False)
            self.taylor_a_0 = nn.Parameter(torch.zeros(1, self.input_dim, self.output_dim))
            self.mask_a_0 = torch.ones(1, self.input_dim, self.output_dim
                                       ).to(device=self.device).requires_grad_(False)
            self.final_bias = torch.zeros(1, self.output_dim).to(self.device).requires_grad_(False)
        else:
            raise ValueError('data_order should be 1 (features) or 2 (features * seq_len)')

        self.to(self.device)

    def forward(self, x):  # x: (batch_size, seq_len, input_dim)
        list_shape = [1 for _ in range(len(x.shape) + 1)]
        list_shape[-1] = self.k  # [1, 1, 1, k]
        y = x.unsqueeze(dim=-1).repeat(tuple(list_shape))

        for iii in range(1, self.k):
            if self.data_order == 1:
                new_values = y[:, :, iii - 1] * y[:, :, 0]
                y = torch.cat((y[:, :, :iii], new_values.unsqueeze(dim=-1),
                               y[:, :, iii+1:]), dim=2)
            elif self.data_order == 2:
                new_values = y[:, :, :, iii - 1] * y[:, :, :, 0]
                y = torch.cat((y[:, :, :, :iii],
                               new_values.unsqueeze(dim=-1),
                               y[:, :, :, iii+1:]), dim=-1)
            else:
                raise ValueError('data_order should be 1 (features) or 2 (features * seq_len)')

        self.taylor_a_0_1.data = self.taylor_a_0_1.data * self.mask_paras
        self.taylor_a_0.data = self.taylor_a_0.data * self.mask_a_0

        y = y.unsqueeze(dim=-1)

        y = y * self.taylor_a_0_1 * self.mask_paras
        y = y.sum(dim=-2)  # k
        y = y + self.taylor_a_0 * self.mask_a_0
        y = y.sum(dim=-2)  # input_dim

        return y

    def prune_(self):
        # make zero
        self.taylor_a_0_1.data = torch.where(
            torch.logical_and(self.taylor_a_0_1.data > -self.threshold, self.taylor_a_0_1.data < self.threshold),
            0, self.taylor_a_0_1.data
        )
        self.taylor_a_0.data = torch.where(
            torch.logical_and(self.taylor_a_0.data > -self.threshold, self.taylor_a_0.data < self.threshold),
            0, self.taylor_a_0.data
        )

        # create mask
        self.mask_paras = torch.abs(self.taylor_a_0_1.data) >= self.threshold
        self.mask_a_0 = torch.abs(self.taylor_a_0.data) >= self.threshold

        self.taylor_a_0_1.data = self.taylor_a_0_1.data * self.mask_paras
        self.taylor_a_0.data = self.taylor_a_0.data * self.mask_a_0

        return self

    def check_pruning_effectiveness(self):
        self.zero_elements = torch.sum(self.taylor_a_0_1.data == 0)
        self.total_elements = self.taylor_a_0_1.numel()
        print(f"Pruning effectiveness: {self.zero_elements}/{self.total_elements} parameters are zeroed out")


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.total_in = self.args.enc_in
        self.hidden = self.args.hidden
        self.total_out = self.args.c_out
        self.k1 = self.args.taylor_terms_order_1
        self.k2 = self.args.taylor_terms_order_2

        if self.KAN_layers == 2:
            self.first_layer = ModelSingle(self.args, input_dim=self.total_in, output_dim=self.hidden,
                                           data_order=2, k=self.k1)
            self.last_layer = ModelSingle(self.args, input_dim=self.hidden, output_dim=self.total_out,
                                          data_order=2, k=self.k2)
        elif self.KAN_layers > 2:
            self.first_layer = ModelSingle(self.args, input_dim=self.total_in, output_dim=self.hidden,
                                           data_order=2, k=self.k1)
            self.middle_layers = nn.ModuleList(
                [ModelSingle(self.args, input_dim=self.hidden, output_dim=self.hidden,
                             data_order=2, k=self.k1) for _ in range(self.KAN_layers - 2)]
            )
            self.last_layer = ModelSingle(self.args, input_dim=self.hidden, output_dim=self.total_out,
                                          data_order=2, k=self.k2)
        else:
            raise ValueError('KAN_layers should >= 2')

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        xx = self.first_layer(x)
        if self.KAN_layers > 2:
            for layer in self.middle_layers:
                xx = layer(xx)
        xxx = self.last_layer(xx)
        return xxx

    def prune_(self):
        self.first_layer.prune_()
        self.last_layer.prune_()
        return self

    def check_pruning_effectiveness(self):
        self.first_layer.check_pruning_effectiveness()
        self.last_layer.check_pruning_effectiveness()
        return self



