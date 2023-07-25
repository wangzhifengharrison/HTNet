import pandas
import os
import shutil

import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
from PIL import Image
import scipy.io
import itertools
from sklearn.metrics import confusion_matrix
    # 绘制混淆矩阵
df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')

m, n = df.shape

print(m,n)

smic_subname = []
samm_subname = []
casme_2_subname = []

for i in range(m):
    dataset_name = df.iloc[i]['dataset']
    if dataset_name == 'casme2' and str(df.iloc[i]['sub']) not in casme_2_subname:
        casme_2_subname.append(str(df.iloc[i]['sub']))
    if dataset_name == 'smic' and  str(df.iloc[i]['sub']) not in smic_subname:
        smic_subname.append(str(df.iloc[i]['sub']))
    if dataset_name == 'samm' and  str(df.iloc[i]['sub']) not in samm_subname:
        samm_subname.append(str(df.iloc[i]['sub']))
print(smic_subname,len(smic_subname))
print(samm_subname,len(samm_subname))
print(casme_2_subname,len(casme_2_subname))


three_dataset_pred_truth = {'011': {'pred': [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 'truth': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '017': {'pred': [1, 0, 0, 0], 'truth': [2, 0, 0, 0]}, 's18': {'pred': [2, 2, 2, 2, 2, 0, 0], 'truth': [2, 2, 2, 2, 2, 0, 0]}, 'sub08': {'pred': [0], 'truth': [0]}, '018': {'pred': [2, 0, 0], 'truth': [2, 0, 0]}, '036': {'pred': [0], 'truth': [0]}, 'sub20': {'pred': [0, 0], 'truth': [0, 0]}, 's19': {'pred': [2, 1], 'truth': [2, 1]}, '019': {'pred': [1], 'truth': [1]}, '021': {'pred': [0, 0], 'truth': [0, 0]}, '006': {'pred': [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '033': {'pred': [2, 0, 0, 0, 0], 'truth': [1, 0, 0, 0, 0]}, 's14': {'pred': [2, 2, 2, 2, 2, 0, 1, 0, 0, 0], 'truth': [2, 2, 2, 2, 2, 1, 1, 0, 0, 0]}, '030': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, 'sub12': {'pred': [2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0], 'truth': [2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0]}, 's02': {'pred': [2, 2, 1, 2, 2, 1], 'truth': [2, 2, 2, 2, 2, 1]}, '015': {'pred': [2, 0, 2], 'truth': [2, 0, 0]}, '009': {'pred': [2, 0, 0, 0], 'truth': [2, 0, 0, 0]}, 'sub05': {'pred': [2, 0, 2, 2, 2, 1], 'truth': [2, 2, 2, 2, 2, 1]}, 'sub16': {'pred': [0, 1, 0], 'truth': [1, 1, 0]}, '014': {'pred': [2, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'truth': [2, 1, 1, 1, 1, 1, 1, 1, 0, 0]}, '016': {'pred': [2, 2, 1, 0, 0], 'truth': [2, 2, 1, 0, 0]}, 'sub25': {'pred': [2, 2, 0, 0, 0], 'truth': [2, 2, 0, 0, 0]}, '013': {'pred': [0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0]}, 'sub07': {'pred': [0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0]}, 's15': {'pred': [2, 2, 0, 0], 'truth': [2, 2, 1, 0]}, 'sub22': {'pred': [0, 0], 'truth': [0, 0]}, 'sub17': {'pred': [2, 0, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '012': {'pred': [2, 0, 0], 'truth': [2, 0, 0]}, 's05': {'pred': [0, 0], 'truth': [2, 0]}, 'sub19': {'pred': [2, 2, 2, 2, 2, 0, 1, 1, 0, 0, 0], 'truth': [2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]}, 's11': {'pred': [2, 1, 1, 1, 0, 0, 0], 'truth': [2, 1, 1, 1, 0, 0, 0]}, 'sub24': {'pred': [2, 0, 0], 'truth': [2, 0, 0]}, '007': {'pred': [1, 1, 1, 1, 1, 1, 1, 1], 'truth': [2, 2, 1, 1, 1, 1, 1, 0]}, 's08': {'pred': [2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '034': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, 'sub06': {'pred': [2, 2, 1, 0], 'truth': [2, 2, 1, 0]}, 's20': {'pred': [0, 2, 2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], 'truth': [2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 'sub23': {'pred': [1, 0, 0, 0, 0, 0, 0, 0], 'truth': [1, 0, 0, 0, 0, 0, 0, 0]}, 'sub14': {'pred': [1, 1, 1], 'truth': [1, 1, 1]}, '010': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '035': {'pred': [2, 0, 0, 0, 0, 0, 0, 0], 'truth': [2, 0, 0, 0, 0, 0, 0, 0]}, '023': {'pred': [0], 'truth': [0]}, '022': {'pred': [1, 1, 0, 0, 0], 'truth': [1, 1, 0, 0, 0]}, '028': {'pred': [2, 0, 0], 'truth': [2, 2, 0]}, 'sub02': {'pred': [2, 2, 2, 2, 0, 0, 0, 0, 0], 'truth': [2, 2, 2, 1, 0, 0, 0, 0, 0]}, '032': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, 's03': {'pred': [0, 0, 0, 2, 2, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 'truth': [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, '026': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0]}, '020': {'pred': [1, 0, 0, 0], 'truth': [1, 1, 0, 0]}, 'sub26': {'pred': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 's12': {'pred': [2, 1, 1, 1, 1, 1, 1, 1, 1], 'truth': [2, 1, 1, 1, 1, 1, 1, 1, 1]}, 'sub15': {'pred': [2, 1, 0], 'truth': [2, 1, 0]}, 'sub01': {'pred': [1, 0, 0], 'truth': [1, 0, 0]}, 'sub13': {'pred': [1, 1], 'truth': [1, 1]}, 'sub03': {'pred': [2, 0, 0, 0, 0], 'truth': [2, 0, 0, 0, 0]}, 'sub04': {'pred': [0, 0], 'truth': [0, 0]}, '031': {'pred': [0], 'truth': [0]}, 'sub11': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, 'sub09': {'pred': [1, 1, 1, 0, 1, 0, 0, 0, 0, 0], 'truth': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}, '037': {'pred': [0], 'truth': [0]}, 's13': {'pred': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'truth': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}, 'sub21': {'pred': [0], 'truth': [0]}, '024': {'pred': [0], 'truth': [0]}, 's01': {'pred': [0, 0, 0, 0, 0, 0], 'truth': [2, 1, 1, 0, 0, 0]}, 's09': {'pred': [2, 2, 2, 1], 'truth': [2, 2, 2, 1]}, 's04': {'pred': [2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2], 'truth': [2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 's06': {'pred': [0, 2, 0, 0], 'truth': [2, 2, 0, 0]}}


smic_pred= []
smic_truth = []
samm_pred= []
samm_truth = []
casme_2_pred= []
casme_2_truth = []

total_three_pred = []
total_three_truth = []
for key in three_dataset_pred_truth:
    total_three_pred.extend(three_dataset_pred_truth[key]['pred'])
    total_three_truth.extend(three_dataset_pred_truth[key]['truth'])
    if key in smic_subname:
        smic_pred.extend(three_dataset_pred_truth[key]['pred'])
        smic_truth.extend(three_dataset_pred_truth[key]['truth'])
    if key in samm_subname:
        samm_pred.extend(three_dataset_pred_truth[key]['pred'])
        samm_truth.extend(three_dataset_pred_truth[key]['truth'])
    if key in casme_2_subname:
        casme_2_pred.extend(three_dataset_pred_truth[key]['pred'])
        casme_2_truth.extend(three_dataset_pred_truth[key]['truth'])

print(len(total_three_truth))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    pylab.show()


cnf_matrix = confusion_matrix(total_three_truth,total_three_pred)
print(cnf_matrix)
attack_types = ['Negative', 'Positive', 'Surprise']
# plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')
plot_confusion_matrix(cnf_matrix, classes=attack_types, title = 'Full', normalize=True)


cnf_matrix_smic = confusion_matrix(smic_truth,smic_pred)
print(cnf_matrix_smic)
attack_types = ['Negative', 'Positive', 'Surprise']
plot_confusion_matrix(cnf_matrix_smic, classes=attack_types, title = 'SMIC', normalize=True)

cnf_matrix_samm = confusion_matrix(samm_truth,samm_pred)
print(cnf_matrix_samm)
attack_types = ['Negative', 'Positive', 'Surprise']
plot_confusion_matrix(cnf_matrix_samm, classes=attack_types, title = 'SAMM', normalize=True)

cnf_matrix_casme_2 = confusion_matrix(casme_2_truth,casme_2_pred)
print(cnf_matrix_casme_2)
attack_types = ['Negative', 'Positive', 'Surprise']
plot_confusion_matrix(cnf_matrix_casme_2, classes=attack_types, title = 'CASME II', normalize=True)

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''
# print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
print('total UF1: ',round(recognition_evaluation(total_three_truth,total_three_pred)[0],4), '| UAR:',round(recognition_evaluation(total_three_truth,total_three_pred)[1],4))
print('SMIC UF1: ',round(recognition_evaluation(smic_truth,smic_pred)[0],4), '| UAR:',round(recognition_evaluation(smic_truth,smic_pred)[1],4))
print('SAMM UF1: ',round(recognition_evaluation(samm_truth,samm_pred)[0],4), '| UAR:',round(recognition_evaluation(samm_truth,samm_pred)[1],4))
print('CASME II UF1: ',round(recognition_evaluation(casme_2_truth,casme_2_pred)[0],4), '| UAR:',round(recognition_evaluation(casme_2_truth,casme_2_pred)[1],4))