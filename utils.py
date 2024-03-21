import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp  #interp函数，用于插值计算，可用于在已知数据点之间进行线性插值或曲线插值，常用于绘制ROC曲线和PR曲线时的插值计算
from sklearn import metrics
import torch
import torch.nn as nn
import dgl

#dis_miRNA_association_matrix.xlsx
def load_data(directory, random_seed):
    D_semSIM = np.loadtxt(directory + '/disease_semantic_sim.txt')
    M_fucSIM = np.loadtxt(directory + '/miRNA_functional_sim.txt')
    M_seqSIM = np.loadtxt(directory + '/miRNA_sequence_sim.txt')
    D_GS_SIM = np.loadtxt(directory + '/dis_GaussianSimilarity.txt')
    M_GS_SIM = np.loadtxt(directory + '/mirna_GaussianSimilarity.txt')

    # SIM = np.loadtxt(directory + '/SIM.txt')

    # file_path = 'G:\\Code\\plachong\\datamd\\relations.csv'  # 将 'your_csv_file.csv' 替换为你的文件路径

########## 原始RNA与疾病关联 ############
    file_path = 'G:\\Code\\gatnew\\data\\relations.csv'  # 将 'your_csv_file.csv' 替换为你的文件路径
    md_all_associations = pd.read_csv(file_path)

##############案例分析
    ########### （1）疾病9 肝细胞癌#########
    # file_path = 'G:\\Code\\gatnew\\case_study\\dis9_Cracinoma_relations.csv'  # 将 'your_csv_file.csv' 替换为你的文件路径
    # md_all_associations = pd.read_csv(file_path)
    ########### （2）疾病3 乳腺肿瘤#########
    # file_path = 'G:\\Code\\gatnew\\case_study\\dis3_Colorectal_relations.csv'  # 将 'your_csv_file.csv' 替换为你的文件路径
    # md_all_associations = pd.read_csv(file_path)
    ########### （3）疾病6 结肠直肠肿瘤#########
    # file_path = 'G:\\Code\\gatnew\\case_study\\dis6_Breast_relations.csv'  # 将 'your_csv_file.csv' 替换为你的文件路径
    # md_all_associations = pd.read_csv(file_path)

    # 统计第三列中0和1的数量
    column_name = 'Value'  # 将 '第三列' 替换为你的列名
    count_0 = len(md_all_associations[md_all_associations[column_name] == 0])
    count_1 = len(md_all_associations[md_all_associations[column_name] == 1])

    print(f'第三列中有 {count_0} 个0 和 {count_1} 个1。')

#     md_all_associations = pd.read_csv(directory + '/relations.csv', names=['miRNA', 'disease', 'label'])
#
# #######################'
#     # 统计第三列中0和1的数量
#     column_name = 'label'  # 将 '第三列' 替换为你的列名
#     count_0 = len(md_all_associations[md_all_associations[column_name] == 0])
#     count_1 = len(md_all_associations[md_all_associations[column_name] == 1])
#
#     print(f'第三列中有 {count_0} 个0 和 {count_1} 个1。')
#########################

    # D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    # D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    # D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    # M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    # M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    # IL = np.loadtxt(directory + '/lncRNA-fuction.txt')
    # md_all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])

    # DataFrame是pandas库中的一个数据结构，用于存储和操作二维表格数据
    # ml_associations = pd.read_csv(directory + '/miRNA-lncRNA.csv', names=['miRNA', 'lncRNA', 'label'])
    # ld_associations = pd.read_csv(directory + '/disease-lncRNA.csv', names=['lncRNA', 'disease', 'label'])
    # D_SSM = (D_SSM1 + D_SSM2) / 2

# 将 D_SSM 赋值给 ID，双重循环遍历 D_SSM 的每一个元素，如果元素的值为0，
# 则将 ID 对应位置的元素替换为 D_GSM 中对应位置的元素
#     weight1 = 0.5
#     weight2 = 0.5
    # 使用加权平均融合相似性矩阵
    # IM = (weight1 * M_fucSIM) + (weight2 * M_seqSIM)

    ID = D_GS_SIM
    IM = M_GS_SIM
#################################################################################
# 筛选miRNA-disease正样本和与正样本数相同的负样本
#loc是Pandas库中的索引操作符，用于根据行标签和列标签进行数据的访问和选择
    known_associations = md_all_associations.loc[md_all_associations['Value'] == 1]
    print('known_associations',len(known_associations))
    unknown_associations = md_all_associations.loc[md_all_associations['Value'] == 0]
    print('unknown_associations',len(unknown_associations))
    # random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)
    random_negative = unknown_associations.sample(n=1 * known_associations.shape[0], random_state=random_seed, axis=0)
    print('random_negative',len(random_negative))
    # 数量n：未知关联数量较多，选择和已知关联数目相同的未知关联组成样本
    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)     # 指针重置
    samples = sample_df.values  # 获得重新编号的新样本
    # 将已知关联和随机负样本合并到一个新的DataFrame sample_df 中，以便在后续的训练或评估中使用
    #
##########################################################################


# # 筛选miRNA-lncRNA和disease-lncRNA已知关联
#     ml_associations1 = ml_associations.loc[ml_associations['label'] == 1]
#     ld_associations1 = ld_associations.loc[ld_associations['label'] == 1]
#     sample_df = known_associations.append(random_negative)
#     #将已知关联和随机负样本合并到一个新的DataFrame sample_df 中，以便在后续的训练或评估中使用


    #对名为 sample_df 的 DataFrame 执行了重置索引的操作，参数 drop=True 表示将旧的索引丢弃，
    # 参数 inplace=True 表示在原始 DataFrame 上进行操作，不创建新的对象。这样做是为了使索引重新编号，并且不保留旧的索引
    # ml_associations1.reset_index(drop=True, inplace=True) #inplace=True 是指将操作应用于原始的 DataFrame
    # ld_associations1.reset_index(drop=True, inplace=True)

    # ml_associations = ml_associations1.values
    # ld_associations = ld_associations1.values

###########案列分析采样###########################
    # sample_df = md_all_associations
    # sample_df.reset_index(drop=True, inplace=True)
    # samples = sample_df.values
    #########################################
    return ID, IM, samples


def load_dataG():
    # pd.read_excel('G:\\Code\\GATGCNML\\DISdata\\dis_gene_association_matrix.xlsx', engine='openpyxl')
    #M_GS + M_fucSIM = MM
    MM = pd.read_excel('G:\\Code\\gatnew\\mgdata\\M_SIM.xlsx', engine='openpyxl')
    MM = torch.tensor(MM.values, dtype=torch.float32)

    # D_GS + D_semSIM = DD
    # DD = pd.read_excel('G:\\Code\\gatnew\\dgdata\\dis_GaussianSimilarity.xlsx', engine='openpyxl')
    DD = pd.read_excel('G:\\Code\\gatnew\\dgdata\\D_SIM.xlsx', engine='openpyxl')
    DD = torch.tensor(DD.values, dtype=torch.float32)

    GG = pd.read_excel('G:\\Code\\gatnew\\dgdata\\gene_GaussianSimilarity.xlsx', engine='openpyxl')
    GG = torch.tensor(GG.values, dtype=torch.float32)

    MA = pd.read_excel('G:\\Code\\gatnew\\mgdata\\miRNA_Gene_association_matrix.xlsx', engine='openpyxl')
    MA = torch.tensor(MA.values, dtype=torch.float32)

    DA = pd.read_excel('G:\\Code\\gatnew\\dgdata\\dis_gene_association_matrix.xlsx', engine='openpyxl')
    DA = torch.tensor(DA.values, dtype=torch.float32)

    # 代谢物

    # MET = pd.read_excel('G:\\Code\\gatnew\\dismetdata\\met_GaussianSimilarity.xlsx', engine='openpyxl')
    # MET = torch.tensor(MET.values, dtype=torch.float32)
    #
    # META = pd.read_excel('G:\\Code\\gatnew\\dismetdata\\dis_met_association_matrix.xlsx', engine='openpyxl')
    # META = torch.tensor(META.values, dtype=torch.float32)

    # DD = torch.tensor(pd.read_excel(r".\dgdata\dis_GaussianSimilarity.xlsx").values, dtype=torch.float32)
    #
    # GG = torch.tensor(pd.read_excel(r".\mgdata\gene_GaussianSimilarity.xlsx").values, dtype=torch.float32)
    #
    # MA = torch.tensor(pd.read_excel(r".\mgdata\miRNA_Gene_association_matrix.xlsx").values, dtype=torch.float32)
    # DA = torch.tensor(pd.read_excel(r".\dgdata\dis_gene_association_matrix.xlsx").values, dtype=torch.float32)



    # MM = torch.tensor(np.loadtxt(r".\mgdata\MM.txt"), dtype=torch.float32)
    #
    # DD = torch.tensor(np.loadtxt(r".\dgdata\DD.txt"), dtype=torch.float32)
    #
    # GG = torch.tensor(np.loadtxt(r".\mgdata\DD.txt"), dtype=torch.float32)
    #
    # MA = torch.tensor(np.loadtxt(r".\data\miRNA-disease association.txt", dtype=int, delimiter="\t"),
    #                  dtype=torch.float32)
    #
    # DA = torch.tensor(np.loadtxt(r".\data\miRNA-disease association.txt", dtype=int, delimiter="\t"),
    #                   dtype=torch.float32)

    MMA = torch.cat((MM, MA), dim=1)
    AGG = torch.cat((MA.T, GG), dim=1)
    MG = torch.cat((MMA, AGG), dim=0)

    DDA = torch.cat((DD, DA), dim=1)
    AGG = torch.cat((DA.T, GG), dim=1)
    DG = torch.cat((DDA, AGG), dim=0)

    # 代谢物
    # DDMETA = torch.cat((DD,META), dim=1)
    # METAMET = torch.cat((META.T, MET), dim=1)
    # DMET = torch.cat((DDMETA, METAMET), dim=0)


    # return MG, DG, DMET
    return MG, DG

# miRNA-disease异质图和miRNA-disease-lncRNA异质图的构建
def build_graph(directory, random_seed):
    ID, IM, samples = load_data(directory, random_seed)

    # miRNA-disease二元异质图
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])  # 在图 g 中添加节点， 添加了 ID.shape[0] + IM.shape[0] 个节点
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    # 创建了一个名为 node_type 的全零张量张量（tensor）,数据类型为 torch.int64。长度（number_of_nodes()）等于图 g 中节点的总数，用来表示图中每个节点的类型
    node_type[: ID.shape[0]] = 1    # 前 ID.shape[0] 个元素设置为 1，表示对应的节点是疾病节点
    g.ndata['type'] = node_type  # 将节点的类型信息存储为 'type' 属性

# 0-374设为疾病节点，并传入特征(将相似性赋值进去)
    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    # 创建一个全0二阶张量d_sim，张量的行数是图 g 中节点的总数，列数是疾病特征的维度
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))  # 用 ID 的值来填充 d_sim 的一部分
    # [: ID.shape[0], :] 表示从 d_sim 中选取所有行（节点）和所有列（特征）
    # : ID.shape[0]只选取前 ID.shape[0] 行,实际上是将 d_sim 矩阵中的前 ID.shape[0] 行提取出来
    # d_sim 是一个二阶张量，即一个矩阵，其中的每一行代表一个节点（通常是疾病节点），而每一列代表一个特征
    # 通过切片操作，将 ID 数据转换为 PyTorch 张量，并将其复制到 d_sim 张量的前 ID.shape[0] 行中，ID.shape[0] 是疾病节点的数量
    g.ndata['d_sim'] = d_sim  # 将 d_sim 张量存储为图 g 的节点数据的一个属性 'd_sim'

# 374-1162设为miRNA节点，并传入特征
    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    # m_sim 中的前面部分保持为零，而 miRNA 节点的特征被填充进去
    # ID.shape[0]+IM.shape[0] 表示 miRNA 节点的数量，先将 NumPy 数组 IM 转换为 PyTorch 张量，选择了图 g 中从疾病节点后面开始的连续行
    g.ndata['m_sim'] = m_sim
    # 将 m_sim 赋值给图 g 的节点数据字典 ndata 中的键 'm_sim'

# 让指针从0开始，原本节点标签从1开始
    # 创建两个列表 disease_ids 和 mirna_ids，这些列表包含了疾病节点和 miRNA 节点的标识号
    disease_ids = list(range(1, ID.shape[0]+1))  # disease_ids 包含从1到 ID.shape[0] 的整数
    # 创建了一个从 1 到 ID.shape[0] 的整数 序列，表示疾病节点的标识号，从 1 开始一直到疾病节点的总数
    mirna_ids = list(range(1, IM.shape[0]+1))  # 这个序列的目的是为了表示 miRNA节点的标识号，从 1 开始一直到 miRNA 节点的总数

# 创建 dis 节点和 miRNA 节点的 id 反向映射字典，将标识号映射到索引。 用于根据 id 查找节点在图中的索引 #构建图 g 中的边（edges）
    # 字典
    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}
    # 创建了两个字典 disease_ids_invmap 和 mirna_ids_invmap，它们用于将疾病和miRNA的原始ID映射到图 g 中节点的索引位置

# 创建了两个列表 sample_disease_vertices 和 sample_mirna_vertices，用于将采样的疾病节点和 miRNA 节点的标识号（ID）映射为图中节点的索引
    # 列表

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]

    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]
    # 遍历 samples 数组的每一行，将原始的疾病ID和miRNA ID 转换成图 g 中节点的索引位置，这些节点索引可以用于在图数据中定位疾病节点和 miRNA 节点

    # g.add_edges(sample_disease_vertices, sample_mirna_vertices,
    #             data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    # samples[:, 2] 是从 samples 数据中提取的边的标签信息，它表示这些边的类别或标签
    # data={'label': torch.from_numpy(samples[:, 2].astype('float32') 表示将边的标签信息以键值对的形式存储在边的数据中，
    # 其中键是 'label'，值是从 samples 数据提取的标签信息
    # 代码调用 g.add_edges 方法，将这些节点索引转换为图 g 中的边
    # 从疾病到miRNA的边和从miRNA到疾病的边。同时，它为每条边添加了一个名为 label 的属性，这个属性存储了每个样本的标签，这些标签是从 samples 数组的第三列提取出来的
    g.readonly()
    # 图 g 设置为只读模式，以防止在后续的操作中对其进行修改

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000) #生成一个均匀分布的从0到1的数组，数组的长度为20000
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

#####################################
    np.savetxt('G:\\Code\\gatnew\\工具\\DUIBIDATA\\teature16_mean_fpr.csv', mean_fpr, delimiter=',')
    np.savetxt('G:\\Code\\gatnew\\工具\\DUIBIDATA\\teature16_mean_tpr.csv', mean_tpr, delimiter=',')
#####################################


    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()






def new_normalization(w):
    m,n = w.size()  # 获取矩阵 w 的行数，通常表示矩阵中的节点数量
    normalized_w = torch.zeros((m, n), dtype=torch.float32)  # (m, m) 的零矩阵
    for i in range(m):
        row_sum = w[i, :].sum()
        if row_sum > 0:
            normalized_w[i, :] = w[i, :] / row_sum

    return normalized_w
