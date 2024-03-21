import warnings
import numpy as np
from train import Train
from utils import plot_auc_curves, plot_prc_curves

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data',
                                                      epochs=800,
                                                      attn_size=64,
                                                      attn_heads=4,
                                                      out_dim=64,
                                                      dropout=0.5,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=1e-4,
                                                      random_seed=2023,
                                                      cuda=True,
                                                      model_type='MNFLMDA')
    plot_auc_curves(fprs, tprs, auc, directory='G:\\Code\\MNFLMDA\\aucresult', name='test_auc')
    plot_prc_curves(precisions, recalls, prc, directory='G:\\Code\\MNFLMDA\\aucresult', name='test_prc')



