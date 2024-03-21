import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl

def load_data(directory, random_seed):

    D_GS_SIM = np.loadtxt(directory + '/dis_GaussianSimilarity.txt')
    M_GS_SIM = np.loadtxt(directory + '/mirna_GaussianSimilarity.txt')


    file_path = 'G:\\Code\\MNFLMDA\\data\\relations.csv'
    md_all_associations = pd.read_csv(file_path)

    ID = D_GS_SIM
    IM = M_GS_SIM

    known_associations = md_all_associations.loc[md_all_associations['Value'] == 1]
    print('known_associations',len(known_associations))
    unknown_associations = md_all_associations.loc[md_all_associations['Value'] == 0]
    print('unknown_associations',len(unknown_associations))
    random_negative = unknown_associations.sample(n=1 * known_associations.shape[0], random_state=random_seed, axis=0)
    print('random_negative',len(random_negative))

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    samples = sample_df.values

    return ID, IM, samples


def load_dataG():
    MM = pd.read_excel('G:\\Code\\MNFLMDA\\mgdata\\M_SIM.xlsx', engine='openpyxl')
    MM = torch.tensor(MM.values, dtype=torch.float32)

    DD = pd.read_excel('G:\\Code\\MNFLMDA\\dgdata\\D_SIM.xlsx', engine='openpyxl')
    DD = torch.tensor(DD.values, dtype=torch.float32)

    GG = pd.read_excel('G:\\Code\\MNFLMDA\\dgdata\\gene_GaussianSimilarity.xlsx', engine='openpyxl')
    GG = torch.tensor(GG.values, dtype=torch.float32)

    MA = pd.read_excel('G:\\Code\\MNFLMDA\\mgdata\\miRNA_Gene_association_matrix.xlsx', engine='openpyxl')
    MA = torch.tensor(MA.values, dtype=torch.float32)

    DA = pd.read_excel('G:\\Code\\MNFLMDA\\dgdata\\dis_gene_association_matrix.xlsx', engine='openpyxl')
    DA = torch.tensor(DA.values, dtype=torch.float32)

    MMA = torch.cat((MM, MA), dim=1)
    AGG = torch.cat((MA.T, GG), dim=1)
    MG = torch.cat((MMA, AGG), dim=0)

    DDA = torch.cat((DD, DA), dim=1)
    AGG = torch.cat((DA.T, GG), dim=1)
    DG = torch.cat((DDA, AGG), dim=0)
    return MG, DG


def build_graph(directory, random_seed):
    ID, IM, samples = load_data(directory, random_seed)

    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type
    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

# 374-1162设为miRNA节点，并传入特征
    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    disease_ids = list(range(1, ID.shape[0]+1))
    mirna_ids = list(range(1, IM.shape[0]+1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]

    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

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
