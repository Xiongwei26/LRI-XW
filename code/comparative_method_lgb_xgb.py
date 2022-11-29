"""
作者：XWEI
日期：2022年10月02日
"""
# Comparison method of lightgbm and xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import KTBoost.KTBoost as KTBoost
import gpboost as gb
import xgboost as xgb
import lightgbm as lgb


def Splicing_data(proteinFeature_L, proteinFeature_R, interaction):
    positive_feature = []
    negative_feature = []
    for i in range(np.shape(interaction)[0]):  
        for j in range(np.shape(interaction)[1]):  
            temp = np.append(proteinFeature_L[i], proteinFeature_R[j])
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)  
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)  

    negative_sample_index = np.random.choice(np.arange(len(negative_feature)), size=len(positive_feature),
                                             replace=False)  
    negative_sample_feature = []
    for i in negative_sample_index:
        negative_sample_feature.append(negative_feature[i])  
    feature = np.vstack((positive_feature, negative_sample_feature))  
    label1 = np.ones((len(positive_feature), 1))  
    label0 = np.zeros((len(negative_sample_feature), 1))  
    label = np.vstack((label1, label0))  
    return feature, label  


proteinFeature_L = pd.read_csv("NEW_DATA/1/812.csv", header=None, index_col=None).to_numpy()  # NumPy数组格式导入
proteinFeature_R = pd.read_csv("NEW_DATA/1/780.csv", header=None, index_col=None).to_numpy()
interaction = pd.read_csv("NEW_DATA/1/Related.csv", header=None, index_col=None).to_numpy()

print(proteinFeature_L.shape)
acc_20_kt = []
precision_20_kt = []
recall_20_kt = []
f1_20_kt = []
AUC_20_kt = []
AUPR_20_kt = []
time = 0
for i in range(20):

    nmn, labels = Splicing_data(proteinFeature_L, proteinFeature_R, interaction)
    GB = gb.GPBoostClassifier()
    GB.fit(nmn, labels.ravel())
    importantFeatures = GB.feature_importances_
    Values = np.sort(importantFeatures)[::-1]  

    feature_number = -importantFeatures
    K1 = np.argsort(feature_number)
    K1 = K1[:400]
    nmn = nmn[:, K1]
    _range = np.max(nmn) - np.min(nmn)
    nmn = (nmn - np.min(nmn)) / _range
    print('--------------End of dimension reduction--------------------------')

    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    AUPR = 0

    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(nmn, labels):
        feature_train, feature_test = nmn[train_index], nmn[test_index]
        target_train, target_test = labels[train_index], labels[test_index]

        # clf = lgb.LGBMClassifier(n_estimators=500, max_depth=15, learning_rate=0.2)

        clf = xgb.XGBClassifier(max_depth=15, learning_rate=0.01,
                                n_estimators=500,
                                objective="binary:logistic", booster='gbtree',
                                n_jobs=3, nthread=3, gamma=1, min_child_weight=1,
                                max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                                reg_alpha=1, reg_lambda=2, scale_pos_weight=1,
                                base_score=0.5)
        clf.fit(feature_train, target_train.ravel())

        y_ = clf.predict(feature_test)
        prob_output = clf.predict_proba(feature_test)
        y_prob_1 = np.arange(0, dtype=float)
        for i in prob_output:
            y_prob_1 = np.append(y_prob_1, i[1])

        acc += accuracy_score(target_test, y_)
        precision += precision_score(target_test, y_)
        recall += recall_score(target_test, y_)
        f1 += f1_score(target_test, y_)

        fpr, tpr, thresholds = roc_curve(target_test, y_prob_1)
        prec, rec, thr = precision_recall_curve(target_test, y_prob_1)
        AUC += auc(fpr, tpr)
        AUPR += auc(rec, prec)
    time = time + 1
    print("-------------CV3 %d  -------------------" % time)
    acc = acc / 5
    precision = precision / 5
    recall = recall / 5
    f1 = f1 / 5
    AUC = AUC / 5
    AUPR = AUPR / 5
    acc_20_kt.append(acc)
    precision_20_kt.append(precision)
    recall_20_kt.append(recall)
    f1_20_kt.append(f1)
    AUC_20_kt.append(AUC)
    AUPR_20_kt.append(AUPR)
    print('-----------------------------------------------------------------------------------')
    # print("accuracy:%.4f" % acc)
    # print("precision:%.4f" % precision)
    # print("recall:%.4f" % recall)
    # print("F1 score:%.4f" % f1)
    # print("AUC:%.4f" % AUC)
    # print("AUPR:%.4f" % AUPR)

    # --------------------boost----------------------------------
    with open("C:/Users/xiongwei/Desktop/NEW_G_K/xgboost/574_xgb.txt", mode="a") as f:
        f.write("xgb___The {} th fold cross validation\n".format(time))
        f.write("precision:{}\n".format(precision))
        f.write("recall:{}\n".format(recall))
        f.write("acc:{}\n".format(acc))
        f.write("f1:{}\n".format(f1))
        f.write("AUC:{}\n".format(AUC))
        f.write("AUPR:{}\n\n".format(AUPR))
        f.write('\n')

print("--------------------------------------循环结束----------------------------------------")
# --------------------------------------------------------------------------------------------------------------------------
with open("C:/Users/xiongwei/Desktop/NEW_G_K/xgboost/574_xgb.txt", mode="a") as f:
    f.write("-------------------------------------Mean value after 20 cycles-----------------------------------------\n")
with open("C:/Users/xiongwei/Desktop/NEW_G_K/xgboost/574_xgb.txt", mode="a") as f:
    f.write("xgb___Mean value after 20 cycles\n")
    f.write("precision:" + str(np.around(np.mean(precision_20_kt), 4)) + "+" + str(
        np.around(np.std(np.array(precision_20_kt)), 4)))
    f.write('\n')
    f.write(
        "recall:" + str(np.around(np.mean(recall_20_kt), 4)) + "+" + str(np.around(np.std(np.array(recall_20_kt)), 4)))
    f.write('\n')
    f.write("accuracy:" + str(np.around(np.mean(acc_20_kt), 4)) + "+" + str(np.around(np.std(np.array(acc_20_kt)), 4)))
    f.write('\n')
    f.write("F1:" + str(np.around(np.mean(f1_20_kt), 4)) + "+" + str(np.around(np.std(np.array(f1_20_kt)), 4)))
    f.write('\n')
    f.write("AUC:" + str(np.around(np.mean(AUC_20_kt), 4)) + "+" + str(np.around(np.std(np.array(AUC_20_kt)), 4)))
    f.write('\n')
    f.write("AUPR:" + str(np.around(np.mean(AUPR_20_kt), 4)) + "+" + str(np.around(np.std(np.array(AUPR_20_kt)), 4)))
    f.write('\n')
    
