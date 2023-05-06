from operator import le, truediv
import pickle
import os
import json
import hashlib
from turtle import position
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import warnings
import csv


def main():
    res_loss={}
    with open('###/AST_loss.json','r') as af:
        AST_res=json.load(af)
    with open('###/CFG_loss.json','r') as cf:
        CFG_res=json.load(cf)
    with open('####/PDG_loss.json','r') as pf:
        PDG_res=json.load(pf)
    for item in PDG_res:
        try:
            # matrix1 = np.matrix(AST_res[item])
            # matrix2 = np.matrix(CFG_res[item])
            # matrix3 = np.matrix(PDG_res[item])
            # matrix4 = np.concatenate((matrix1,matrix2,matrix3), axis=1)
            matrix4 = AST_res[item]+CFG_res[item]+PDG_res[item]
            # matrix4 = PDG_res[item]+CFG_res[item]
            res_loss[item] = matrix4
        except:
            continue
    print('before filter len:\t%d'%(len(res_loss)))
    for outliers_fraction in np.arange(0.05, 0.35, 0.05): # 0.05 - 0.5
        print('====> outliers_fraction:\t%.3f'%(outliers_fraction))
        final_keep_slice_path, iforest = iforest_filter(res_loss, outliers_fraction)
        print(len(final_keep_slice_path))
        revise_label = list(set(list(res_loss.keys())) - set(final_keep_slice_path))
        with open('#####/revise_%.2f.txt'%(outliers_fraction),'a') as w1p:
           for item in final_keep_slice_path:
               item="####/PDG_input/"+item
               w1p.write(item+'\n')
        w1p.close()
        with open('####/revise_%.2f.txt'%(outliers_fraction),'a') as w1p:
           for item in revise_label:
               if int(item[0]) == 0:
                   item="####/PDG_input/"+'1\'_CVE'+item.split('CVE')[-1]
               else:
                   item="######/PDG_input/"+'0\'_CVE'+item.split('CVE')[-1]
               w1p.write(item+'\n')
        w1p.close()

def iforest_filter(keep_slice_dict, outliers_fraction):
    vul_keep_slices = []
    slice_name_list = []
    for func_ in keep_slice_dict.keys():
        vul_keep_slices.append(keep_slice_dict[func_])
        slice_name_list.append(func_)
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=len(vul_keep_slices), random_state=rng, contamination=outliers_fraction)
    clf.fit(X=vul_keep_slices)

    scores_pred = clf.decision_function(vul_keep_slices)
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)  #####

    final_keep_slice_path = []
    for i, _score in enumerate(scores_pred):
        if _score > threshold:
            final_keep_slice_path.append(slice_name_list[i])
    return final_keep_slice_path, clf

if __name__ == '__main__':
    main()