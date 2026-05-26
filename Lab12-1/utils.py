import numpy as np
import pandas as pd
import torch
import random
from sklearn.cluster import KMeans
import numbers

# 计算各组 TPR
def groupTPR(p_predict, y_true, group_label, ind):
    group_set = set(group_label)
    if len(group_set) > 5 and isinstance(list(group_set)[0], numbers.Number):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(group_label.reshape(-1,1))
        group_label = group_label[ind.int()]
        group_label = kmeans.predict(group_label.reshape(-1,1))
        group_set = set(group_label)
    else:
        group_label = group_label[ind.int()]
        group_set = set(group_label)

    group_tpr = []
    for group in group_set:
        group_true_ind = np.array([a==1 and b==group for a, b in zip(y_true, group_label)])
        cur_tpr = p_predict[group_true_ind,:][:,1].mean()
        if not np.isnan(cur_tpr):
            group_tpr.append(cur_tpr)
    return group_tpr

# 计算各组 TNR
def groupTNR(p_predict, y_true, group_label, ind):
    group_set = set(group_label)
    if len(group_set) > 5 and isinstance(list(group_set)[0], numbers.Number):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(group_label.reshape(-1,1))
        group_label = group_label[ind.int()]
        group_label = kmeans.predict(group_label.reshape(-1,1))
        group_set = set(group_label)
    else:
        group_label = group_label[ind.int()]
        group_set = set(group_label)

    group_tnr = []
    for group in group_set:
        group_true_ind = np.array([a==0 and b==group for a, b in zip(y_true, group_label)])
        cur_tnr = p_predict[group_true_ind,:][:,0].mean()
        if not np.isnan(cur_tnr):
            group_tnr.append(cur_tnr)
    return group_tnr

# 生成对抗样本
def counter_sample(X_raw, ind, related_attr, scaler):
    X_new = X_raw.copy()
    attr_candid = list(set(X_raw[related_attr]))
    attr_new = random.choices(attr_candid, k=ind.shape[0])
    X_new.loc[ind, related_attr] = attr_new
    X_new = pd.get_dummies(X_new)
    X_new = X_new.sort_index(axis=1)
    X_new = scaler.transform(X_new.iloc[ind])
    return torch.FloatTensor(X_new)

# 计算敏感属性与相关属性相关性
def cal_correlation(X_raw, sens_attr, related_attr):
    X_src = pd.get_dummies(X_raw[sens_attr])
    X_relate = pd.get_dummies(X_raw[related_attr])
    correffics = []
    for i in range(len(X_src.keys())):
        for j in range(len(X_relate.keys())):
            corr = abs(X_src[X_src.keys()[i]].corr(X_relate[X_relate.keys()[j]]))
            correffics.append(corr)
    return sum(correffics)