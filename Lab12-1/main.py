import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import MetricFrame
import utils
import argparse

# 命令行参数
parser = argparse.ArgumentParser(description='FairML')
parser.add_argument("--epoch", default=2, type=int)
parser.add_argument("--pretrain_epoch", default=1, type=int)
parser.add_argument("--method", default="base", type=str,
                    choices=['base','corre','groupTPR','learn','remove','learnCorre'])
parser.add_argument("--dataset", default="adult", type=str, choices=['adult','pokec','compas','law'])
parser.add_argument("--s", default="sex", type=str)
parser.add_argument("--related", nargs='*', type=str, default=None)
parser.add_argument("--r_weight", nargs='*', type=float, default=None)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weightSum", default=0.3, type=float)
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model", default='MLP', type=str, choices=['MLP','LR','SVM'])
args = parser.parse_args()

# 随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print('beta: {}, weightSum: {}'.format(args.beta, args.weightSum))

# 加载数据
if args.dataset == 'adult':
    data = fetch_openml(data_id=1590)
    header = list(data.data.columns)
    if args.method == 'remove':
        for attr in args.related:
            header.remove(attr)
        data.data = data.data[header]
    X = pd.get_dummies(data.data)
    X = X.sort_index(axis=1)
    y_true = ((data.target == '>50K') * 1).values
    n_classes = y_true.max() + 1
    data_frame = data.data
    sensitive_attr = data_frame[args.s]

    for relate in args.related:
        coef = utils.cal_correlation(data.data, args.s, relate)
        print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))
    data.data['target'] = data.target
    for relate in args.related:
        coef = utils.cal_correlation(data.data, 'target', relate)
        print('coefficient between {} and {} is: {}'.format('target', relate, coef))

if args.dataset == 'compas':
    df = pd.read_csv("Lab12-1/data/Processed_Compas.csv")
    data_frame = df.copy()

    # 标签列检查（CSV 中是 is_recid）
    if 'is_recid' in df.columns:
        y_true = df['is_recid'].values
    else:
        raise KeyError("找不到标签列 'is_recid'，请检查 CSV 列名: {}".format(df.columns.tolist()))

    # 敏感属性检查
    if args.s in df.columns:
        sensitive_attr = df[args.s]
    else:
        raise KeyError("找不到敏感属性列 '{}'，可用列: {}".format(args.s, df.columns.tolist()))

    # 候选特征（去掉敏感属性和标签）
    cand_feats = [c for c in df.columns if c not in [args.s, 'is_recid']]

    # 若未传 related，计算并打印与敏感属性的相关系数（供你选择 top4）
    if not args.related:
        print("Computing correlations between '{}' and other features:".format(args.s))
        corrs = []
        for feat in cand_feats:
            coef = utils.cal_correlation(df, args.s, feat)
            corrs.append((feat, coef))
        corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in corrs_sorted:
            print(f"{feat}: {coef}")
        print("如果要继续训练并使用相关属性公平性，请选择 4 个属性作为 --related 重新运行 main.py。")
    else:
        for relate in args.related:
            coef = utils.cal_correlation(data_frame, args.s, relate)
            print('coefficient between {} and {} is: {}'.format(args.s, relate, coef))

    # 特征矩阵构造（去掉标签和敏感属性）
    X = df.drop(['is_recid', args.s], axis=1)
    X = pd.get_dummies(X)
    X = X.sort_index(axis=1)
    n_classes = len(np.unique(y_true))

    # 统一 related / r_weight 的变量（后续训练使用这两个变量）
# 若未在数据分支中定义 related_attrs/related_weights（如 adult 分支未定义），确保变量存在
    related_attrs = globals().get('related_attrs', args.related or [])
    related_weights = globals().get('related_weights', args.r_weight or [])
    if len(related_weights) < len(related_attrs):
        if len(related_weights) == 0:
            related_weights = [0.0] * len(related_attrs)
        else:
            related_weights = related_weights + [related_weights[-1]] * (len(related_attrs) - len(related_weights))

# 数据划分
indict = np.arange(sensitive_attr.shape[0])
X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
    X, y_true, indict, test_size=0.5, stratify=y_true, random_state=7
)
processed_X_train = X_train
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 自定义数据集
class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super().__init__(*tensors)
    def _df_to_tensor(self, df):
        if isinstance(df, np.ndarray):
            return torch.from_numpy(df).float()
        return torch.from_numpy(df.values).float()

train_data = PandasDataSet(X_train, y_train, ind_train)
test_data = PandasDataSet(X_test, y_test, ind_test)
train_loader = DataLoader(train_data, batch_size=320, shuffle=True, drop_last=True)
print('# training samples:', len(train_data))
print('# batches:', len(train_loader))

# 分类器
class Classifier(nn.Module):
    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_class),
        )
    def forward(self, x):
        return self.network(x)

class Classifier_lr(nn.Module):
    def __init__(self, n_features, n_class=2):
        super().__init__()
        self.linear = nn.Linear(n_features, n_class)
    def forward(self, x):
        return self.linear(x)

# SVM损失
def loss_SVM(result, truth, model):
    truth[truth==0] = -1
    result = result.squeeze()
    weight = model.linear.weight.squeeze()
    loss = torch.mean(torch.clamp(1 - truth*result, min=0))
    loss += 0.1 * torch.mean(weight**2)
    return loss

# 初始化模型
n_features = X.shape[1]
n_hid = 72 if args.dataset == 'pokec' else 32
if args.model == 'MLP':
    clf = Classifier(n_features=n_features, n_hidden=n_hid, n_class=n_classes)
elif args.model == 'LR':
    clf = Classifier_lr(n_features=n_features, n_class=n_classes)
elif args.model == 'SVM':
    assert n_classes == 2
    clf = Classifier_lr(n_features=n_features, n_class=1)
else:
    raise NotImplementedError
clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)

# 预训练
def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        loss.backward()
        optimizer.step()
    return clf

# 对抗扰动训练
def Perturb_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        for related_attr, related_weight in zip(related_attrs, related_weights):
            x_new = utils.counter_sample(data.data, ind.int(), related_attr, scaler)
            p_y_new = clf(x_new)
            p_stack = torch.stack((p_y[:,1], p_y_new[:,1]), dim=1)
            p_order = torch.argsort(p_stack, dim=-1)
            cor_loss = torch.square(p_stack[:, p_order[:,1].detach()] - p_stack[:, p_order[:,0]]).mean()
            loss += cor_loss * related_weight
        loss.backward()
        optimizer.step()
    return clf

# 相关性消除训练
def CorreErase_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        for related_attr, related_weight in zip(related_attrs, related_weights):
            selected = [related_attr in col for col in processed_X_train.keys()]
            mean_x = x[:, selected].mean(dim=0, keepdim=True)
            diff = x[:, selected] - mean_x
            corr = torch.mean(diff * (p_y - p_y.mean(dim=0)).unsqueeze(1), dim=0)
            cor_loss = torch.sum(torch.abs(corr))
            loss += cor_loss * related_weight
        loss.backward()
        optimizer.step()
    return clf

# 群体公平训练
def Gfair_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        for related_attr, related_weight in zip(related_attrs, related_weights):
            group_tpr = utils.groupTPR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            group_loss = torch.square(max(group_tpr) - min(group_tpr))
            loss += group_loss * related_weight
        loss.backward()
        optimizer.step()
    return clf

# 训练入口
if args.model != 'SVM':
    clf_criterion = nn.CrossEntropyLoss()
else:
    clf_criterion = loss_SVM

related_attrs = args.related
related_weights = args.r_weight

# 预训练
for _ in range(args.pretrain_epoch):
    clf.train()
    clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)

# 主训练
for _ in range(args.epoch):
    clf.train()
    if args.method == 'base':
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)
    elif args.method == 'conterfactual':
        clf = Perturb_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    elif args.method == 'corre':
        clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    elif args.method == 'groupTPR':
        clf = Gfair_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)

# 测试
clf.eval()
with torch.no_grad():
    pre_test = clf(test_data.tensors[0])
    if args.model != 'SVM':
        y_pred = pre_test.argmax(dim=1).numpy()
    else:
        y_pred = (pre_test > 0).squeeze().numpy().astype(int)

# 评估
sens_test = sensitive_attr.iloc[ind_test]
gm = MetricFrame(metrics=metrics.accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=sens_test)
print('Average accuracy:', gm.overall)
print(gm.by_group)