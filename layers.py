import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, dropout, slope):
        super(GATLayer, self).__init__()
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.G = G
        self.slope = slope
        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': F.elu(h)}


    def forward(self, G):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)
        return self.G.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)


def add_gaussian_noise(x, mean=0, std=0.1):
    noise = torch.randn(x.size()) * std + mean
    noisy_x = x + noise
    return noisy_x



class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_weights = torch.tanh(self.W(x))
        attention_weights = self.V(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights

class AutoencoderWithAttentionM(nn.Module):
    def __init__(self):
        super(AutoencoderWithAttentionM, self).__init__()
        self.encoderM = nn.Sequential(
            nn.Linear(1134, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.attention = AttentionLayer(128, 128)
        self.decoderM = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1134),
            nn.Tanh()
        )

    def forward(self, x):
        encodedM = self.encoderM(x)
        attention_weights = self.attention(encodedM)
        weighted_encoded = attention_weights * encodedM
        _ = self.decoderM(weighted_encoded)
        return weighted_encoded
class AutoencoderWithAttentionD(nn.Module):
    def __init__(self):
        super(AutoencoderWithAttentionD, self).__init__()
        self.encoderD = nn.Sequential(
            nn.Linear(720, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.attention = AttentionLayer(128, 128)   # 注意力层
        self.decoderD = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 720),
            nn.Tanh()
        )

    def forward(self, x):
        encodedD = self.encoderD(x)
        attention_weights = self.attention(encodedD)
        weighted_encoded = attention_weights * encodedD
        _ = self.decoderD(weighted_encoded)
        return weighted_encoded

