import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tensorflow.python.keras.utils import np_utils
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score

from xgboost import XGBClassifier
from time import time
import os
import psutil


itime = time()
def define_model():
    
    ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(200, ), name='Protein_a')
    p11 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_1', kernel_regularizer=l2(0.01))(input_1)
    p11 = Dropout(0.2)(p11)
    
    p12 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_2', kernel_regularizer=l2(0.01))(p11)
    p12 = Dropout(0.2)(p12)
    
    p13= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_3', kernel_regularizer=l2(0.01))(p12)
    p13 = Dropout(0.2)(p13)
    
    p14= Dense(64, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_4', kernel_regularizer=l2(0.01))(p13)
    p14 = Dropout(0.2)(p14)
    
    ########################################################"Channel-2" ########################################################
    
    input_2 = Input(shape=(200, ), name='Protein_b')
    p21 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_1', kernel_regularizer=l2(0.01))(input_2)
    p21 = Dropout(0.2)(p21)

    p22 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_2', kernel_regularizer=l2(0.01))(p21)
    p22 = Dropout(0.2)(p22)

    p23= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_3', kernel_regularizer=l2(0.01))(p22)
    p23 = Dropout(0.2)(p23)

    p24= Dense(64, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_4', kernel_regularizer=l2(0.01))(p23)
    p24 = Dropout(0.2)(p24)
   


    ##################################### Merge Abstraction features ##################################################
    
    merged = concatenate([p14, p24], name='merged_protein1_2')
    
    ##################################### Prediction Module ##########################################################
    
    pre_output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_1')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(pre_output)
    pre_output = Dense(16, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)


    
    pre_output=Dropout(0.2)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)
    model = Model(inputs=[input_1, input_2], outputs=output)
   
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

##################################### Load Positive and Negative Dataset ##########################################################
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

num = pd.read_csv("nmn1.csv", header=None, index_col=None).to_numpy()  
label = pd.read_csv("labels1.csv", header=None, index_col=None).to_numpy()


# proteinFeature_L = pd.read_csv("NEW_DATA_686/1/812.csv", header=None, index_col=None).to_numpy()  
# proteinFeature_R = pd.read_csv("NEW_DATA_686/1/780.csv", header=None, index_col=None).to_numpy()
# interaction = pd.read_csv("NEW_DATA_686/1/Related.csv", header=None, index_col=None).to_numpy()

# print(proteinFeature_L.shape)
# nmn, labels = Splicing_data(proteinFeature_L, proteinFeature_R, interaction)
# import gpboost as gb
# GB = gb.GPBoostClassifier(importance_type='gain')
# GB.fit(nmn, labels.ravel())
# importantFeatures = GB.feature_importances_
# Values = np.sort(importantFeatures)[::-1]  # 排序
#
# feature_number = -importantFeatures
# K1 = np.argsort(feature_number)
# K1 = K1[:400]
# nmn = nmn[:, K1]

# _range = np.max(nmn) - np.min(nmn)
# nmn = (nmn - np.min(nmn)) / _range
#
# xgb_fpr=pd.DataFrame(nmn)
# xgb_tpr=pd.DataFrame(labels)
#
# xgb_fpr.to_csv('nmn.csv',header=False, index=False)
# xgb_tpr.to_csv('labels.csv',header=False, index=False)
#
# Trainlabels = labels

X1_train = num[:, :200]
X2_train = num[:, 200:]



##################################### Five-fold Cross-Validation ##########################################################
    
kf=StratifiedKFold(n_splits=5)


accuracy1 = []
specificity1 = []
sensitivity1 = []
precision1=[]
recall1=[]

m_coef=[]
dnn_fpr_list=[]
dnn_tpr_list=[]
dnn_auc_list = []
o=0
max_accuracy=float("-inf")
dnn_fpr=None
dnn_tpr=None

for train, test in kf.split(num,label):
    global model
    model=define_model()
    o=o+1

    model.fit([X1_train[train],X2_train[train]],label[train],epochs=50,batch_size=64,verbose=1)
    y_test = label[test]
    y_score = model.predict([X1_train[test], X2_train[test]])

################################Intermediate Layer prediction (Abstraction features extraction)######################################
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('merged_protein1_2').output)
intermediate_output_p1 = intermediate_layer_model.predict([X1_train,X2_train])  
p_merge=pd.DataFrame(intermediate_output_p1)    
X_train_feat=pd.concat((p_merge,pd.DataFrame(pd.DataFrame(label))),axis=1,ignore_index=True)

Train = X_train_feat
Train=Train.sample(frac=1)
X=Train.iloc[:,0:128].values
y=Train.iloc[:,128:].values

extracted_df = X_train_feat

scaler=RobustScaler()
X=scaler.fit_transform(X)


##################################### Five-fold Cross-Validation ##########################################################

kf=StratifiedKFold(n_splits=5)

# AUPRs = []
accuracy = []
specificity = []
sensitivity = []
precision=[]
recall=[]
m_coef=[]

auc_list=[]
xgb_fpr_list=[]
xgb_tpr_list=[]
o=0
max_accuracy=float("-inf")
xgb_fpr=None
xgb_tpr=None
# PREs = np.array([])
# ACCs = np.array([])
# RECs = np.array([])
# AUCs = np.array([])
# AUPRs = np.array([])
# F1s = np.array([])
for j in range(5):
    for train, test in kf.split(X, y):
        o = o + 1
        model = XGBClassifier(n_estimators=100)

        hist = model.fit(X[train], y[train], eval_set=[(X[test], y[test])])
        pre_label = model.predict(X[test])
        y_score = model.predict_proba(X[test])
        y_test = np_utils.to_categorical(y[test])
#         acc = accuracy_score(y_test[:, 0], pre_label)
#         rec = recall_score(y_test[:, 0], pre_label)
#         f1 = f1_score(y_test[:, 0], pre_label)
#         pre = precision_score(y_test[:, 0], pre_label)
        fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
        pre_, rec_, _ = precision_recall_curve(y_test[:, 0], y_score[:, 0])
        au = metrics.roc_auc_score(y_test, y_score)
        auc_list.append(au)
        coef=matthews_corrcoef(y_test.argmax(axis=1), y_score.argmax(axis=1), sample_weight=None)
        m_coef.append(coef)
        cm1=confusion_matrix(y_test.argmax(axis=1), y_score.argmax(axis=1))

        acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
        spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
        sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
        pre_=cm1[1,1]/(cm1[1,1]+cm1[0,1])
        rec_=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    # aupr = auc(rec_, pre_)
        sensitivity.append(sens)
        specificity.append(spec)
        accuracy.append(acc)
        precision.append(pre_)
        recall.append(rec_)
    # AUPRs.append(aupr)
        xgb_fpr_list.append(fpr)
        xgb_tpr_list.append(tpr)
        if max_accuracy < acc:
            max_accuracy = acc
            xgb_fpr = fpr
            xgb_tpr = tpr

xgb_fpr=pd.DataFrame(xgb_fpr)
xgb_tpr=pd.DataFrame(xgb_tpr)

# xgb_fpr.to_csv('fprdnn_xgb.csv',header=False, index=False)
# xgb_tpr.to_csv('tprdnn_xgb.csv',header=False, index=False)

mean_acc=np.mean(accuracy)
std_acc=np.std(accuracy)
var_acc=np.var(accuracy)
print("Accuracy:"+str(mean_acc)+" ± "+str(std_acc))
print("Accuracy_Var:"+str(mean_acc)+" ± "+str(var_acc))
mean_spec=np.mean(specificity)
std_spec=np.std(specificity)
print("Specificity:"+str(mean_spec)+" ± "+str(std_spec))
mean_sens=np.mean(sensitivity)
std_sens=np.std(sensitivity)
print("Sensitivity:"+str(mean_sens)+" ± "+str(std_sens))
mean_prec=np.mean(precision)
std_prec=np.std(precision)
print("Precison:"+str(mean_prec)+" ± "+str(std_prec))
mean_rec=np.mean(recall)
std_rec=np.std(recall)
print("Recall:"+str(mean_rec)+" ± "+str(std_rec))
mean_coef=np.mean(m_coef)
std_coef=np.std(m_coef)
print("MCC:"+str(mean_coef)+" ± "+str(std_coef))
std_auc=np.std(auc_list)
print("AUC:"+str(np.mean(auc_list))+" ± "+str(std_auc))

print('One time 5-Folds computed. Time: {}秒, mem: {}MB'.format((time() - itime), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
