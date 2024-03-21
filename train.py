import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold       # 用于实现K折交叉验证的类，将数据集划分为训练集和验证集
from sklearn import metrics     #用于评估模型性能和计算各种指标的函数和类
import pickle
from Get_low_dimension_feature import get_low_feature
from sklearn.metrics import auc

from utils import load_data, load_dataG, build_graph, weight_reset      # 分别是用于加载数据的,构建图结构的,重置模型权重的函数
from model import AEGATMF

# directory数据目录的路径，slope指LeakyReLU 斜率
def Train(directory, epochs, attn_size, attn_heads, out_dim, dropout, slope, lr, wd, random_seed, cuda, model_type):
    random.seed(random_seed)
    np.random.seed(random_seed)  # 为了确保实验的可重现性，我们需要在每次运行时得到相同的随机结果
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)

    # if cuda:
    #     context = torch.device('cuda:0')
    # else:
    #     context = torch.device('cpu')
    context = torch.device('cpu')
    # g为miRNA和disease图，g0为miRNA、disease和lncRNA图
    #samples是一个包含miRNA和disease之间关联的样本的NumPy数组
    #ID,IM,IL是疾病，mirna，lrna综合相似度矩阵
    # g, g0, disease_vertices, mirna_vertices, ID, IM, IL, samples, ml_associations, ld_associations = build_graph(directory, random_seed)
    g,disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    # 将样本数据转换为DataFrame格式，便于后续处理和分析。

    print('## vertices(nodes):', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())
   # print('## lncrna nodes: ', torch.sum(g0.ndata['type'] == 2).numpy())

    #该代码行用于统计图中疾病节点的数量。具体而言，g.ndata['type']表示图g中节点的类型信息，其中节点类型为1表示疾病节点。
    # 通过torch.sum(g.ndata['type'] == 1)可以计算出图中节点类型为1（疾病节点）的节点数量。
    # .numpy()用于将结果转换为NumPy数组格式进行打印输出。
    g.to(context)
    # 将图数据转移到指定的计算设备上

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

###################### JUZHENGFENGJIE #######################################
# # Define a dictionary to store low-dimensional features
#     low_dim_features = {}
#     # A = pd.read_excel('G:\\Code\\gatnew\\data\\dis_miRNA_association_matrix.xlsx', engine='openpyxl')
#     A = np.load('G:\\Code\\gatnew\\data\\miRNA-disease association.npy')
#     print('start Get low-dimensional features U and V')
#     U, V = get_low_feature(32, 0.01, A)  # pow(10, -4)
#     low_dim_features['U'] = U
#     low_dim_features['V'] = V
#     mirna_feature = low_dim_features['U']
#     dis_feature = low_dim_features['V']
#     print("dis_feature:", dis_feature.shape)
#     print("mirna_feature:", mirna_feature.shape)
#     print('finished Get low-dimensional features')
#######################################################################


# 设置五折交叉验证
    i = 0
    n = 5
    kf = KFold(n_splits=n, shuffle=True, random_state=random_seed)  #shuffle=True表示在划分折时打乱数据顺序

    for train_idx, test_idx in kf.split(samples[:, 2]):  # 通过kf.split(samples[:, 2])生成训练集和测试集的索引
        i += 1
        print('Training for Fold', i)  #在每个折中，代码输出当前所处的折数，例如"Training for Fold 1"表示正在进行第一折的训练
# 将训练集的指针标记为1，其余为0
        # 标出训练集
        samples_df['train'] = 0
        # 首先，创建一个名为 train 的新列，并将该列的所有元素初始化为0。这一列将用于标记样本是否属于训练集。
        samples_df['train'].iloc[train_idx] = 1
        # 通过索引 train_idx 将训练集中的样本对应的 train 列的值设置为1
        # 使用索引train_idx选择训练集样本，并将相应的'train'列的值设为1。其他样本的'train'列值保持为0

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))
        # 将NumPy数组转换为PyTorch张量

        edge_data = {'train': train_tensor}
        # 将其存储在edge_data字典中的'train'键下

# 对两个异质图的训练边进行标记
#         g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        #通过使用 g.edges[disease_vertices, mirna_vertices]，我们可以选取 miRNA 到 disease 的边，
        # 并使用 update() 方法来更新这些边的属性数据，将 edge_data 中的数据更新到选定的边上
        #g0.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        #g0.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        # 由于图的边的添加训练的原因，g0和g不包含lncRNA节点的边的标记相同，然后分别根据训练边构建子图

        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        # rain_eid 是一个包含训练集边的边的ID（边索引）的列表
        # 从图 g 中筛选出训练集的边，并创建一个仅包含这些训练集边的子图 g_train
        # g_train是根据训练集中的边创建的训练图，它保留了与这些边相关联的节点
        # g_train0 = g0.edge_subgraph(train_eid, preserve_nodes=True)
        # g_train.copy_from_parent() dgl0.4之后的版本没有这个语句

# 训练标签和测试标签用于最后对比
        label_train = g_train.edata['label'].unsqueeze(1)
        # 从训练图 g_train 的边数据（edata）中提取标签（通常是边的权重或标记），然后转换为列向量
# xiugai
        label_train = label_train.reshape(-1)  # 重新塑造为一维数组
        # 在这段代码中，label_train 是从训练图 g_train 的边数据中提取的标签，通过 unsqueeze(1) 将其转换为列向量。
        src_train, dst_train = g_train.all_edges()  # 获取所有训练图 g_train 中的边
        # 分别是训练图 g_train 的所有边的源节点和目标节点，将被用作模型的输入特征

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        # 从图数据中筛选出测试集的边（edges），然后获取这些边的源节点（src_test）和目标节点（dst_test）
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)
        # g.edges[test_eid]: 这一部分用于从图 g 中获取测试集的边。test_eid 通常包含了测试集边的边标识符
#xiugai
        # 将目标标签的形状从 [814, 1] 调整为 [814]
        label_test = label_test.squeeze()

        print('## Training edges:', len(train_eid))    #打印出训练集边的数量，即变量 train_eid 的长度
        print('## Testing edges:', len(test_eid))

        MG, DG = load_dataG()

        if model_type == 'AEGATMF':
            model = AEGATMF(
                           G=g_train,#G=g_train0,
                           #meta_paths_list=['md', 'dm', 'ml', 'dl'],
                           feature_attn_size=attn_size,  # 64
                           num_heads=attn_heads,
                           num_diseases=ID.shape[0],  # 是指疾病的数量，它的值等于 ID 数组的行数（shape[0]）
                           num_mirnas=IM.shape[0],
                           # num_lncrnas=IL.shape[0],
                           #d_sim_dim=ID.shape[1],   #维度
                           #m_sim_dim=IM.shape[1],
                           low_feature_dim=32,
                           out_dim=out_dim,    # 64
                           dropout=dropout,
                           slope=slope,
                           )


        model.apply(weight_reset)       # 是一个自定义的函数，用于对模型的参数进行初始化
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # loss = nn.BCELoss()   #定义了二分类任务的损失函数，这里使用了二分类交叉熵损失
        loss = F.binary_cross_entropy


        best_auc = 0.0
        #训练模型的主要部分
        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                # score_train = model(g_train0, g0, src_train, dst_train)   #调用模型进行前向传播，得到训练集的预测结果
                # score_train = model(g_train, src_train, dst_train, MG, DG, mirna_feature, dis_feature)
                score_train = model(g_train, src_train, dst_train, MG, DG)
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad() #清零优化器的梯度
                loss_train.backward()  #对损失进行反向传播，计算模型参数的梯度
                optimizer.step()  #执行优化器的参数更新，更新模型的参数。


            #模型的验证，在验证阶段，不需要进行梯度计算和参数更新
            model.eval()
            with torch.no_grad():   #用于在验证阶段关闭梯度计算，以减少内存消耗
                # score_val = model(g, g0, src_test, dst_test)
                # score_val = model(g, src_test, dst_test, MG, DG, mirna_feature, dis_feature)
                score_val = model(g, src_test, dst_test, MG, DG)
                loss_val = loss(score_val, label_test)


            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
            # 将训练集的预测结果从GPU上移动到CPU，并将其转换为NumPy数组。squeeze()函数用于去除维度为1的维度，以得到一个一维数组。
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            #pred_val是一个列表，其中存储了对验证集样本的预测结果
            # 将预测结果 score_val_cpu 转换为二分类结果，根据阈值0.5将概率值转换为类别标签0或1
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time() #记录当前时间，以便计算代码的执行时间


            #条件(epoch + 1) % 10 == 0是用于控制打印的频率，每10个epoch打印一次结果
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

            # 保存最好模型
            if val_auc > best_auc:
                best_auc = val_auc
                # 保存最佳模型
                best_model = model.state_dict()
                with open(f'best_model_fold_{i}.pkl', 'wb') as f:
                # with open('G:\\Code\\gatnew\\best_model\\best_model_fold_{i}.pkl', 'wb') as f:
                    pickle.dump(best_model, f)

    # 创建一个空列表来存储每一折的 fpr 和 tpr 数据
    fpr_list = []
    tpr_list = []
   ######加载最好model
    for i in range(n):  # 假设有10折交叉验证
        with open(f'best_model_fold_{i+1}.pkl', 'rb') as f:
            best_model = pickle.load(f)
        # 设置模型状态为最佳模型
        model.load_state_dict(best_model)

        ###测试
        model.eval()
        with torch.no_grad():
            # score_test = model(g, g0, src_test, dst_test)
            # score_test = model(g, src_test, dst_test, MG, DG, mirna_feature, dis_feature)
            score_test = model(g, src_test, dst_test, MG, DG)

####################### 保存得分 ###############
        df = pd.DataFrame({'case_study': score_test})
        df.to_excel('G:\\Code\\gatnew\\工具\\result\\score.xlsx', index=False)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)


        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result