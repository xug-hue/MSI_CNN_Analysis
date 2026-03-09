import keras
import os
import csv
import numpy as np
import pandas as pd
import datatable as dt
from sklearn.model_selection import train_test_split
from tensorflow.keras import *
import shap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##  MSI数据加载 (34840, 92064)
Data_DIR = "MSI/data/MSINorm-all.csv"
my_table = dt.fread(Data_DIR, sep=",", header=False)  ## datatable格式读取文件
HR2MSI_data = my_table.to_numpy()  ## datatable格式转np数组，保存int数据
print(HR2MSI_data.shape)

##  CNN模型加载
MODEL_FILE = "msi_1dcnn_best_model.36-0.99.h5"
model = keras.models.load_model(MODEL_FILE)
print(model.summary())

##  预测结果加载，只加载预测分类,读出类标签1,2,3 (34840,)
Predicted_result_labels = \
    dt.fread('MSI/data/MSIbiaozhuNPArray_3class_CNNPrediction.csv', sep=",", header=False).to_numpy()[:, 0].astype(int)
print(Predicted_result_labels[10000], Predicted_result_labels.dtype)

##  各个数据结构的创建
shap_heatmap_nonorm_sumsample = np.zeros((HR2MSI_data.shape[1], 3))  # 总热力图是每个分类所有shap值热力图的和,(92064, 3)
print(Predicted_result_labels.shape, shap_heatmap_nonorm_sumsample.shape)

np.random.seed(1)
## shap解释值计算过程
# 背景数据选择1：随机采样，随机取1000
# background = HR2MSI_data[np.random.choice(HR2MSI_data.shape[0], 1000, replace=False)]
# 背景数据选择2：分层采样，按照三个类的比例选样本100
background, _, _, _ = train_test_split(HR2MSI_data, Predicted_result_labels, train_size=90,
                                     stratify=Predicted_result_labels)
# print(background)
# 背景数据选择3：只随机采样一个类的样本

# 改变数据维度，类似训练模型时候，增加一个维度，数据要变成(参数数量, 1)的形状
background_expanded = np.expand_dims(background, axis=-1)

# DeepExplainer开始计算
explainer = shap.DeepExplainer((model.input, model.output), background_expanded)

# 对每个样本计算特征shap重要性值的热度图，并叠加到该样本分类的总热度图当中
for sampleNumi in range(HR2MSI_data.shape[0]):  # HR2MSI_data.shape[0]
    if sampleNumi % 1000 == 0:
        print(sampleNumi)
    # 读出第i个样本的谱数据吗湖月【哦方法
    spectrum_to_explain = HR2MSI_data[[sampleNumi]]

    # 改变数据维度，类似训练模型时候，增加一个维度，数据要变成(参数数量, 1)的形状
    spectrum_to_explain_expanded = np.expand_dims(spectrum_to_explain, axis=-1)

    # DeepExplainer开始计算每个样本的解释
    raw_shap_explanations = explainer.shap_values(spectrum_to_explain_expanded, check_additivity=True)
    shap_heatmap = raw_shap_explanations[0, :, 0, Predicted_result_labels[sampleNumi] - 1]
    shap_heatmap = np.maximum(shap_heatmap, 0)  # 将重要性负值变成0
    '''
    # dt保存csv dt方法无法压缩数据
    frame = dt.Frame(shap_heatmap / np.max(shap_heatmap))
    frame.to_csv("D:/shap each sample/dt" + str(sampleNumi) + ".csv", header=False)
    '''
    # 归一化后数据存为pd dataframe，再压缩保存
    pd_data = pd.DataFrame(shap_heatmap / np.max(shap_heatmap))
    # 将数据保存到一个新的CSV文件中
    pd_data.to_csv("D:/shap each sample/" + str(sampleNumi) + ".csv", compression='gzip', index=False, header=False)

    # 总热力图计算,非正则化叠加，
    shap_heatmap_nonorm_sumsample[:, Predicted_result_labels[sampleNumi] - 1] += shap_heatmap

# 保存总热力图_没有正则化数据到文件
with open("MSI/result/HR2MSI/shap_heatmap_nonorm_sumsample.csv", 'w', encoding='utf-8', newline="") as f:
    for i in range(len(shap_heatmap_nonorm_sumsample)):
        csv_write = csv.writer(f)
        csv_write.writerow(shap_heatmap_nonorm_sumsample[i])
