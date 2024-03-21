import warnings
import numpy as np
from train import Train
# from case_study import Train
from utils import plot_auc_curves, plot_prc_curves  # 从个名为utils.py的模块中导入了自定义模块plot_auc_curves和plot_prc_curves


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 将警告信息设置为忽略

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data',
                                                      epochs=800,
                                                      attn_size=64,     # 64注意力机制的大小
                                                      attn_heads=4,     # 8注意力机制的头数
                                                      out_dim=64,      # 64
                                                      dropout=0.5,          # 0.5
                                                      slope=0.2,            # 0.2   0 到 1 之间，常见的取值包括 0.01 和 0.2
                                                      lr=0.001,             #0.001
                                                      wd=1e-4,              # 1e-2, 1e-3, 1e-4, 1e-5
                                                      random_seed=2023,
                                                      cuda=True,
                                                      model_type='AEGATMF')  # GATMDA_only_attn   GATMDA_without_attn

    # plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc')
    plot_auc_curves(fprs, tprs, auc, directory='G:\\Code\\gatnew\\aucresult', name='test_auc')
    # plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc')
    plot_prc_curves(precisions, recalls, prc, directory='G:\\Code\\gatnew\\aucresult', name='test_prc')



