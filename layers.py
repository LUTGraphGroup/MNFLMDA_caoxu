import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, dropout, slope):
        super(GATLayer, self).__init__()
        #feature_attn_size（注意力机制中的特征大小
        # G 图被用于过滤出疾病节点和 miRNA 节点
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
# lambda关键字：它用于创建一个匿名函数，也被称为Lambda函数，nodes：这是Lambda函数的参数，表示图中的一个节点，nodes.data['type']：这部分表示从节点nodes中获取属性'type'的值。节点数据通常是一个字典，包含了节点的各种属性
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.G = G
        self.slope = slope  # LeakyReLU激活函数的负斜率
        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        #self.m_fc是一个线性层，将输入特征转换为指定大小的特征向量，它将miRNA节点的输入特征（'m_sim'）映射到指定大小的特征向量，特征大小由feature_attn_size指定。
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()#一个自定义的方法，用于初始化网络参数

# 用于初始化神经网络层的参数。在该方法中，使用Xavier初始化（nn.init.xavier_normal_）对权重进行初始化
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')  # 计算了 ReLU 激活函数的增益值
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)  # self.m_fc.weight 表示神经网络层 self.m_fc 的权重参数
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)  # Xavier是一种常用的权重初始化方法，旨在确保神经网络的权重在前向传播和反向传播过程中保持稳定正态初始化，gain参数指定了初始化的增益值

# 边的注意力 #计算每条边的注意力分数
    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        # 获取了源节点 (edges.src) 和目标节点 (edges.dst) 的特征向量 z,然后，它将这两个特征向量逐元素相乘，得到一个新的向量 a
        # 使用 torch.sum 函数对 a 沿着维度1（每条边的维度）求和，得到每条边的未归一化注意力分数 a
        # 最后，使用 Leaky ReLU 激活函数（F.leaky_relu）对未归一化的注意力分数 a 进行修正，并添加到边的数据字典中，用键 'e' 表示
        # .unsqueeze(1) 操作是将每条边的未归一化注意力分数 a 变形为一个列向量
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

# 定义的一个消息传递函数，用于在消息传递的过程中将消息传递给目标节点
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}
        # 获取了每条边的源节点 (edges.src) 的特征向量 z 和边的注意力分数 e,它返回一个字典，其中包含两个键值对

    def reduce_func(self, nodes):
    #节点聚合函数，节点聚合函数，用于在消息传递的过程中将接收到的消息进行聚合,用于将从相邻节点接收到的信息进行聚合
        # alpha注意力系数  # 获取了从相邻节点接收到的消息，这些消息保存在 nodes.mailbox 中
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # alpha 是一个注意力系数,用来衡量每条边的重要性
        # 对接收到的边的注意力权重 'e' 进行 softmax 归一化，以获得每条边的权重系数 alpha。这个步骤确保了注意力分数之和等于1，以便将注意力集中在相关性较高的邻居节点上
        # 首先对接收到的边的注意力权重'e'进行 softmax 归一化，以获得每条边的权重系数alpha
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        # nodes.mailbox['z'] 包含了每个节点收到的来自相邻节点的特征向量,将相邻节点的特征根据其注意力权重进行加权求和
        # 对每个节点收到的来自相邻节点的特征向量进行加权求和。这将得到每个节点的聚合特征向量 h
        # 将每条边的源节点特征向量'z'与对应的权重系数相乘，并对它们进行求和，得到 目标节点 的聚合特征向量h

        return {'h': F.elu(h)}
        # 聚合特征向量 h 通常会经过一个非线性激活函数（在这个代码中使用了 ELU 激活函数）以产生节点的最终表示。这个表示包含了节点及其相邻节点的信息


    def forward(self, G):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        # self.G.apply_nodes 这行代码用于在图中的一组节点上应用一个函数，以更新或操作这些节点的特征
        # self.G: 这是一个图（Graph）对象，表示了一个图数据结构。在该图中，包含了一些节点和它们之间的边
        # apply_nodes: 这是 DGL 库中的一个函数，用于在一组指定的节点上应用某个函数..self.disease_nodes: 这是一个用于筛选节点的条件。
        # lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}这是一个 lambda 函数，它接受一个节点集合 nodes 作为输入，
        # 然后返回一个字典，字典中包含节点特征的更新。在这里，我们将更新节点的特征 'z'。更新的方式是：首先获取节点的原始特征 'd_sim'，
        # 然后应用一个线性变换 self.d_fc，接着应用一个 dropout 操作 self.dropout，最终将得到更新后的特征 'z'.
        # 对疾病节点的特征进行线性变换，其中 self.d_fc 是一个线性层，将输入的疾病特征转换为具有指定大小的特征向量，并且应用了 dropout 操作以减少过拟合
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        self.G.apply_edges(self.edge_attention)
        # 当调用 self.G.apply_edges(self.edge_attention) 时，它会对图中的每一条边调用 self.edge_attention 函数，将每条边作为输入，然后返回每条边的注意力权重
        # 使用 edge_attention方法,以计算每条边的注意力权重.将注意力权重'e'计算并添加到边的数据中
        self.G.update_all(self.message_func, self.reduce_func)
        # update_all: 这是 DGL 库中的一个函数，用于在整个图上执行消息传递和聚合操作
        # self.message_func: 用于定义如何在图中的每条边上生成消息。这个函数将被应用在每一条边上，以计算需要传递给目标节点的消息内容
        # self.reduce_func: 用于定义如何在目标节点上聚合从邻居节点接收到的消息。这个函数将被应用在每个节点上，以计算节点的新特征
        # 通过update_all方法进行消息传递和聚合操作
        return self.G.ndata.pop('h')
        #最后从图的节点数据中提取出聚合后的节点特征 'h' 并返回

class MultiHeadGATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge
        # 循环执行num_heads次，每次创建一个新的GATLayer实例并将其添加到self.heads中。
        # 这样就创建了一个包含多个GATLayer对象的列表
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        #64*8=512
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)




##########################zibinamaqi################################################


# # 加载数据
#
# def load_data():
#
#     MM = torch.tensor(np.loadtxt(r".\data\MM.txt"), dtype=torch.float32)
#     DD = torch.tensor(np.loadtxt(r".\data\DD.txt"), dtype=torch.float32)
#
#     A = torch.tensor(np.loadtxt(r".\data\miRNA-disease association.txt", dtype=int, delimiter="\t"),
#                      dtype=torch.float32)
#
#     HM = torch.cat((MM, A), dim=1)
#     HD = torch.cat((A.T, DD), dim=1)
#     H = torch.cat((HM, HD), dim=0)
#
#     return H


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

# 定义自编码器模型
class AutoencoderWithAttentionM(nn.Module):
    def __init__(self):
        super(AutoencoderWithAttentionM, self).__init__()
        # self.noise_layer = GaussianNoise(stddev)  # 添加高斯噪声层
        self.encoderM = nn.Sequential(
            nn.Linear(1134, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.attention = AttentionLayer(128, 128)   # 注意力层   # 作用：通过改变权重，来筛选有用特征
        # self.attention = AttentionLayer(32, 32)
        self.decoderM = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1134),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.noise_layer(x)  # 添加高斯噪声
        encodedM = self.encoderM(x)
        attention_weights = self.attention(encodedM)
        weighted_encoded = attention_weights * encodedM
        _ = self.decoderM(weighted_encoded)
        return weighted_encoded

        # _ = self.decoderM(encodedM)
        # return encodedM

class AutoencoderWithAttentionD(nn.Module):
    def __init__(self):
        super(AutoencoderWithAttentionD, self).__init__()
        # self.noise_layer = GaussianNoise(stddev)  # 添加高斯噪声层
        self.encoderD = nn.Sequential(
            nn.Linear(720, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.attention = AttentionLayer(128, 128)   # 注意力层
        # self.attention = AttentionLayer(32, 32)
        self.decoderD = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 720),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.noise_layer(x)  # 添加高斯噪声
        encodedD = self.encoderD(x)
        attention_weights = self.attention(encodedD)
        weighted_encoded = attention_weights * encodedD
        _ = self.decoderD(weighted_encoded)
        return weighted_encoded
        # _ = self.decoderD(encodedD)
        # return encodedD


# class AutoencoderWithAttentionMET(nn.Module):
#     def __init__(self):
#         super(AutoencoderWithAttentionMET, self).__init__()
#         # self.noise_layer = GaussianNoise(stddev)  # 添加高斯噪声层
#         self.encoderMET = nn.Sequential(
#             nn.Linear(1054, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32)
#         )
#         self.attention = AttentionLayer(32, 32)  # 注意力层
#         self.decoderMET = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         # x = self.noise_layer(x)  # 添加高斯噪声
#         encodedMET = self.encoderMET(x)
#         attention_weights = self.attention(encodedMET)
#         weighted_encoded = attention_weights * encodedMET
#         decodedMET = self.decoderMET(weighted_encoded)
#         return decodedMET
