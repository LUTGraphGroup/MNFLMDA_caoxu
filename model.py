import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *



class AEGATMF(nn.Module):
    def __init__(self, G,  feature_attn_size, num_heads, num_diseases, num_mirnas,
                 low_feature_dim, out_dim, dropout, slope):
        super(AEGATMF, self).__init__()
        self.G = G
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.low_feature_dim = low_feature_dim
        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope)
        self.heads = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.m_fc = nn.Linear(feature_attn_size * num_heads + 128, out_dim)
        self.d_fc = nn.Linear(feature_attn_size * num_heads + 128, out_dim)

        self.h_fc = nn.Linear(out_dim, out_dim)
        self.InnerProductDecoder = InnerProductDecoder()

        self.AEAM = AutoencoderWithAttentionM()
        self.AEAD = AutoencoderWithAttentionD()

    def forward(self, G, diseases, mirnas, M, D):
        h_agg0 = self.gat(G)


        dh_agg0 = h_agg0[:374]
        mh_agg0 = h_agg0[374:]
        mirna0 = mh_agg0
        disease0 = dh_agg0


        decodedM = self.AEAM(M)

        decodedM = decodedM[:788]

        decodedD = self.AEAD(D)
        # decodedD = new_normalization(decodedD)
        decodedD = decodedD[:374]


        # AE+GAT
        h_d = torch.cat((disease0, decodedD), dim=1)
        h_m = torch.cat((mirna0, decodedM), dim=1)  # 288


        h_m = self.dropout(F.elu(self.m_fc(h_m)))   # self.m_fc（512，64）   512+32=544
        h_d = self.dropout(F.elu(self.d_fc(h_d)))


        h = torch.cat((h_d, h_m), dim=0)    # # (1162，64）
        h = self.dropout(F.elu(self.h_fc(h)))

        h_diseases = h[diseases]
        h_mirnas = h[mirnas]

        predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)

        return predict_score


# 内积解码器
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mirnas):
        x = torch.mul(h_diseases, h_mirnas).sum(1)
        x = torch.reshape(x, [-1])
        outputs = F.sigmoid(x)
        return outputs

