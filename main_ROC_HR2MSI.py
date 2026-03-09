import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
import csv
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data = load_data_csv()

    DATA_DIR = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/predictresult.csv"
    # 打开predictresult数据文件，前三列是三个类的置信概率值，第四列是数据真实分类标签
    my_table = dt.fread(DATA_DIR, sep=",", header=False)  ## datatable格式读取文件
    data = my_table.to_numpy()      ## datatable格式转np数组，保存int数据
    print(data.shape)
    # 转换分类标签成为onehot
    binary_y = label_binarize(data[:,3], classes=[1, 2, 3])
    # 设置种类
    n_classes = binary_y .shape[1]
    print(n_classes)



    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_y[:, i], data[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    '''Compute micro-average ROC curve and ROC area（方法二）
    首先，对于一个测试样本：1）标签只由0和1组成，1的位置表明了它的类别（可对应二分类问题中的‘’正’’），
    0就表示其他类别（‘’负‘’）；2）要是分类器对该测试样本分类正确，则该样本标签中1对应的位置在概率矩阵P中的值是大于0对应的位置的概率值的。
    基于这两点，将标签矩阵L和概率矩阵P分别按行展开，转置后形成两列，这就得到了一个二分类的结果。所以，此方法经过计算后可以直接得到最终的ROC曲线。
    '''
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_y.ravel(), data[:, 0:3].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ''' Compute macro-average ROC curve and ROC area（方法一）
    每种类别下，都可以得到m个测试样本为该类别的概率（矩阵P中的列）。所以，根据概率矩阵P和标签矩阵L中对应的每一列，
    可以计算出各个阈值下的假正例率（FPR）和真正例率（TPR），从而绘制出一条ROC曲线。这样总共可以绘制出n条ROC曲线。
    最后对n条ROC曲线取平均，即可得到最终的ROC曲线。
    '''
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Plot all ROC curves
    lw = 2  # 线宽
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["micro"]),
             color='b', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["macro"]),
             color='g', linestyle=':', linewidth=4)

    inferno = plt.get_cmap('inferno')
    colors = inferno(np.array([0, 0.5, 1]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:.4f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([-0.05, 1])
    # plt.ylim([0, 1.05])
    plt.xlim([-0.01, 0.1])
    plt.ylim([0.9, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to 3 class')
    plt.legend(loc="lower right")
    plt.show()

