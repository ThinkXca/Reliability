import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. 定义TLSTM模型 ===

class TLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, fc_dim, dropout=0.5):
        super(TLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim

        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Ui = nn.Linear(hidden_dim, hidden_dim)
        self.bi = nn.Parameter(torch.zeros(hidden_dim))

        self.Wf = nn.Linear(input_dim, hidden_dim)
        self.Uf = nn.Linear(hidden_dim, hidden_dim)
        self.bf = nn.Parameter(torch.ones(hidden_dim))

        self.Wo = nn.Linear(input_dim, hidden_dim)
        self.Uo = nn.Linear(hidden_dim, hidden_dim)
        self.bo = nn.Parameter(torch.zeros(hidden_dim))

        self.Wc = nn.Linear(input_dim, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, hidden_dim)
        self.bc = nn.Parameter(torch.zeros(hidden_dim))

        self.W_decomp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_decomp = nn.Parameter(torch.ones(hidden_dim))

        self.fc = nn.Linear(hidden_dim, fc_dim)
        self.classifier = nn.Linear(fc_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def map_elapse_time(self, t):
        return 1 / torch.log(t + 2.7183)

    def forward(self, x, time):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        all_outputs = []

        for i in range(seq_len):
            x_t = x[:, i, :]                     # [batch, input_dim]
            t_t = time[:, i, 0].unsqueeze(1)    # [batch, 1]

            T = self.map_elapse_time(t_t).expand(-1, self.hidden_dim)
            c_st = torch.tanh(self.W_decomp(c_t) + self.b_decomp)
            c_st_dis = T * c_st
            c_t = c_t - c_st + c_st_dis

            i_t = torch.sigmoid(self.Wi(x_t) + self.Ui(h_t) + self.bi)
            f_t = torch.sigmoid(self.Wf(x_t) + self.Uf(h_t) + self.bf)
            o_t = torch.sigmoid(self.Wo(x_t) + self.Uo(h_t) + self.bo)
            c_hat = torch.tanh(self.Wc(x_t) + self.Uc(h_t) + self.bc)

            c_t = f_t * c_t + i_t * c_hat
            h_t = o_t * torch.tanh(c_t)

            fc_out = self.dropout(F.relu(self.fc(h_t)))
            output = self.classifier(fc_out)
            all_outputs.append(output)

        all_outputs = torch.stack(all_outputs, dim=1)
        return all_outputs[:, -1, :]