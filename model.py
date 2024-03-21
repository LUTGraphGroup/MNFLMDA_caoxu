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

        # AE + GAT + MF
        # self.m_fc = nn.Linear(feature_attn_size * num_heads + 128 + low_feature_dim, out_dim)  #64*4+32
        # self.d_fc = nn.Linear(feature_attn_size * num_heads + 128 + low_feature_dim, out_dim)

        # GAT + MF
        # self.m_fc = nn.Linear(feature_attn_size * num_heads + low_feature_dim, out_dim)
        # self.d_fc = nn.Linear(feature_attn_size * num_heads + low_feature_dim, out_dim)

        # AE + GAT
        # self.m_fc = nn.Linear(feature_attn_size * num_heads + 128, out_dim)
        self.d_fc = nn.Linear(feature_attn_size * num_heads + 128, out_dim)

        # GAT
        self.m_fc = nn.Linear(feature_attn_size * num_heads, out_dim)
        # self.d_fc = nn.Linear(feature_attn_size * num_heads, out_dim)

        self.h_fc = nn.Linear(out_dim, out_dim)
        #self.predict = nn.Linear(out_dim * 2, 1)
        # self.BilinearDecoder = BilinearDecoder(feature_size=64)
        self.InnerProductDecoder = InnerProductDecoder()
# self.m_fc, self.d_fc, self.h_fc, 和 self.predict 是神经网络的不同层，通常是线性（全连接）层，用于学习不同的特征表示和最终的预测。
# nn.Linear 是 PyTorch 中的线性层，它执行输入数据和权重矩阵之间的矩阵乘法，然后添加偏差。在这里，它们用于将输入数据变换为不同的表示。
# feature_attn_size, num_heads, low_feature_dim, 和 out_dim 是层中的参数。这些参数控制每个线性层的输入和输出维度。
# 通常，feature_attn_size 和 num_heads 可能与注意力机制（比如多头注意力）有关，而 low_feature_dim 和 out_dim 控制特征的维度。
# 最终的目标是使用这些层来构建一个神经网络，它将输入数据传递到模型中，经过一系列线性变换 （通过 nn.Linear 层），
# 然后进行非线性变换（通过激活函数，如 ReLU），最终通过 self.predict 层产生一个输出，通常用于回归或分类任务。

##################### zibianmaqi ###########################
        self.AEAM = AutoencoderWithAttentionM()
        self.AEAD = AutoencoderWithAttentionD()
        # self.AEMET = AutoencoderWithAttentionMET()


    # def forward(self, G, G0, diseases, mirnas):
    # def forward(self, G, diseases, mirnas, M, D ,mirna_feature, dis_feature):
    def forward(self, G, diseases, mirnas, M, D):
        h_agg0 = self.gat(G)  # feature_attn_size * num_heads
        # disease0 = h_agg0[:374]
        # mirna0 = h_agg0[374:]

        dh_agg0 = h_agg0[:374]
        mh_agg0 = h_agg0[374:]
        mirna0 = mh_agg0
        disease0 = dh_agg0


########################## zibianmaqi #####################
        decodedM = self.AEAM(M)
        # decodedM = new_normalization(decodedM)
        decodedM = decodedM[:788]

        decodedD = self.AEAD(D)
        # decodedD = new_normalization(decodedD)
        decodedD = decodedD[:374]

    # # 代谢物的
    #     decodedDMET = self.AEMET(DMET)
    #     decodedDMET = decodedDMET[:374]
        # weight1 = 0.5
        # weight2 = 0.5
    #
    #     weight11 = 0.3
    #     weight22 = 0.4
    #     weight3 = 0.3
    #     # 使用加权平均融合相似性矩阵
    #     disease = (weight1 * decodedD) + (weight2 * decodedDMET)

    #     mirna0 = (weight1 * decodedM) + (weight2 * mh_agg0)
    #     disease0 = (weight1 * decodedD) + (weight2 * dh_agg0)
    #     disease0 = (weight11 * decodedD) + (weight22 * dh_agg0) + (weight3 * decodedDMET)

        # h_d = torch.cat((disease0, self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        # # self.G.ndata['d_sim'][:self.num_diseases]从图 self.G 中选择了前 self.num_diseases 个节点的特征
        # # d_sim行，num_diseases列dim=1 表示在列维度上进行拼接
        # h_m = torch.cat((mirna0, self.G.ndata['m_sim'][self.num_diseases:]), dim=1)


        # h_d = torch.cat((disease0, decodedD), dim=1)
        # h_m = torch.cat((mirna0, decodedM), dim=1)
        # # h_d = disease0
        # # h_m = mirna0
        #
        # h_m = self.dropout(F.elu(self.m_fc(h_m)))      # 788 32 # （383,64）
        # h_d = self.dropout(F.elu(self.d_fc(h_d)))       # （495,64）

    ######################### 加入juzhenfenjie ###########################
        # dis_feature = torch.from_numpy(dis_feature).float()
        # mirna_feature = torch.from_numpy(mirna_feature).float()
################################################################
        # AE + GAT + MF  AE输出为128维
        # h_d = torch.cat((disease0, decodedD, dis_feature), dim=1) #256+128+32 （374，416）
        # h_m = torch.cat((mirna0, decodedM, mirna_feature), dim=1)  #（788，416）

        # GAT+MF
        # h_d = torch.cat((disease0, dis_feature), dim=1)
        # h_m = torch.cat((mirna0, mirna_feature), dim=1)  # 288

        # AE+GAT
        h_d = torch.cat((disease0, decodedD), dim=1)
        # h_m = torch.cat((mirna0, decodedM), dim=1)  # 288

        # GAT
        # h_d = disease0
        h_m = mirna0

        h_m = self.dropout(F.elu(self.m_fc(h_m)))   # self.m_fc（512，64）   512+32=544
        h_d = self.dropout(F.elu(self.d_fc(h_d)))
    ###############################################################

        h = torch.cat((h_d, h_m), dim=0)    # # (1162，64）
        h = self.dropout(F.elu(self.h_fc(h)))



        h_diseases = h[diseases]
        #（3251，32） # disease中有重复的疾病名称;(17376,64)？？？？？
        # 从张量 h 中选择特定的行，其中 diseases 是一个包含行索引的列表或张量。这种操作通常用于从张量中提取特定样本或数据点
        h_mirnas = h[mirnas]      # （3251，32）  # (17376,64)

        predict_score = self.InnerProductDecoder(h_diseases, h_mirnas)

        return predict_score

# # 双线性解码器
# class BilinearDecoder(nn.Module):
#     def __init__(self, feature_size):
#         super(BilinearDecoder, self).__init__()
#
#         # self.activation = ng.Activation('sigmoid')  # 定义sigmoid激活函数
#         # 获取维度为(embedding_size, embedding_size)的参数矩阵，即论文中的Q参数矩阵
#
#         self.W = Parameter(torch.randn(feature_size, feature_size))
#
#     def forward(self, h_diseases, h_mirnas):
#         h_diseases0 = torch.mm(h_diseases, self.W)
#         h_mirnas0 = torch.mul(h_diseases0, h_mirnas)
#         # h_mirnas0 = h_mirnas.tanspose(0,1)
#         # h_mirnsa0 = torch.mm(h_diseases0, h_mirnas0)
#         h0 = h_mirnas0.sum(1)
#         h = torch.sigmoid(h0)
#         h = h.unsequence(1)
#         return h

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

