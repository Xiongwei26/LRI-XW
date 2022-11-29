from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import KTBoost.KTBoost as KB
from dimension_reduction import KPCA, LLE, dim_NMF
from sklearn.ensemble import AdaBoostClassifier

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
    label1 = np.ones((len(positive_feature), 1)
    label0 = np.zeros((len(negative_sample_feature), 1)) 
    label = np.vstack((label1, label0)) 
    return feature, label  


proteinFeature_L = pd.read_csv("NEW_DATA/1/812.csv", header=None, index_col=None).to_numpy()  
proteinFeature_R = pd.read_csv("NEW_DATA/1/780.csv", header=None, index_col=None).to_numpy()
interaction = pd.read_csv("NEW_DATA/1/Related.csv", header=None, index_col=None).to_numpy()

print(interaction.shape)

acc_20_kt = []
precision_20_kt = []
recall_20_kt = []
f1_20_kt = []
AUC_20_kt = []
AUPR_20_kt = []


times = 1
for i in range(20):
    nmn, labels = Splicing_data(proteinFeature_L, proteinFeature_R, interaction)
    model = AdaBoostClassifier()
    model.fit(nmn, labels)
    importantFeatures = model.feature_importances_
    Values = np.sort(importantFeatures)[::-1]  

    feature_number = -importantFeatures
    K1 = np.argsort(feature_number)
    K1 = K1[:400]
    nmn = nmn[:, K1]
    _range = np.max(nmn) - np.min(nmn)
    nmn = (nmn - np.min(nmn)) / _range

    acc_kt = 0
    precision_kt = 0
    recall_kt = 0
    f1_kt = 0
    AUC_kt = 0
    AUPR_kt = 0

    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(nmn, labels):
        feature_train, feature_test = nmn[train_index], nmn[test_index]
        target_train, target_test = labels[train_index], labels[test_index]

        # KTB
        clf = KB.BoostingClassifier(n_estimators=400, max_depth=9, learning_rate=0.1, n_components=100,
                                    base_learner="combined", kernel="rbf", update_step='hybrid')
        clf.fit(feature_train, target_train.ravel())
        y_ = clf.predict(feature_test)
        prob_output = clf.predict_proba(feature_test)
        y_prob_1 = np.arange(0, dtype=float)
        for x in prob_output:
            y_prob_1 = np.append(y_prob_1, x[1])

        acc_kt += accuracy_score(target_test, y_)
        precision_kt += precision_score(target_test, y_)
        recall_kt += recall_score(target_test, y_)
        f1_kt += f1_score(target_test, y_)

        fpr_kt, tpr_kt, thresholds = roc_curve(target_test, y_prob_1)
        prec_kt, rec_kt, thr = precision_recall_curve(target_test, y_prob_1)
        AUC_kt += auc(fpr_kt, tpr_kt)
        AUPR_kt += auc(rec_kt, prec_kt)
        # -----------------------------------------------------------------------------

    acc_kt = acc_kt / 5
    precision_kt = precision_kt / 5
    recall_kt = recall_kt / 5
    f1_kt = f1_kt / 5
    AUC_kt = AUC_kt / 5
    AUPR_kt = AUPR_kt / 5


    acc_20_kt.append(acc_kt)
    precision_20_kt.append(precision_kt)
    recall_20_kt.append(recall_kt)
    f1_20_kt.append(f1_kt)
    AUC_20_kt.append(AUC_kt)
    AUPR_20_kt.append(AUPR_kt)


    # --------------------boost----------------------------------
    with open("/data/xiongw/AUC_AUPR//812_Adaboost.txt", mode="a") as f:
        f.write("ktboost___Five-fold cross-validation {} times\n".format(times))
        f.write("precision:{}\n".format(precision_kt))
        f.write("recall:{}\n".format(recall_kt))
        f.write("acc:{}\n".format(acc_kt))
        f.write("f1:{}\n".format(f1_kt))
        f.write("AUC:{}\n".format(AUC_kt))
        f.write("AUPR:{}\n\n".format(AUPR_kt))
        f.write('\n')
    times = times + 1

# --------------------------------------------------------------------------------------------------------------------------

with open("/data/xiongw/AUC_AUPR//812_Adaboost.txt", mode="a") as f:
    f.write("-------------------------------------The mean after 20 cycles-----------------------------------------\n")
with open("/data/xiongw/AUC_AUPR//812_Adaboost.txt", mode="a") as f:
    f.write("KTboost___The mean after 20 cycles\n")
    f.write("precision:" + str(np.around(np.mean(precision_20_kt), 4)) + "+" + str(
        np.around(np.std(np.array(precision_20_kt)), 4)))
    f.write('\n')
    f.write("recall:" + str(np.around(np.mean(recall_20_kt), 4)) + "+" + str(np.around(np.std(np.array(recall_20_kt)), 4)))
    f.write('\n')
    f.write("accuracy:" + str(np.around(np.mean(acc_20_kt), 4)) + "+" + str(np.around(np.std(np.array(acc_20_kt)), 4)))
    f.write('\n')
    f.write("F1:" + str(np.around(np.mean(f1_20_kt), 4)) + "+" + str(np.around(np.std(np.array(f1_20_kt)), 4)))
    f.write('\n')
    f.write("AUC:" + str(np.around(np.mean(AUC_20_kt), 4)) + "+" + str(np.around(np.std(np.array(AUC_20_kt)), 4)))
    f.write('\n')
    f.write("AUPR:" + str(np.around(np.mean(AUPR_20_kt), 4)) + "+" + str(np.around(np.std(np.array(AUPR_20_kt)), 4)))
    f.write('\n')




