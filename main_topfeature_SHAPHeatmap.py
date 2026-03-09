import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datatable as dt

if __name__ == "__main__":

    file = "MSI/data/mz list.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datax = my_table.to_numpy()

    topFeaturesShapIndices = np.array([1, 1, 1])
    topFeaturesShapIndices[0] = np.where(datax == np.array([502.975]))[0]
    topFeaturesShapIndices[1] = np.where(datax == np.array([741.541]))[0]
    topFeaturesShapIndices[2] = np.where(datax == np.array([798.54]))[0]
    print(topFeaturesShapIndices)

    # 初始化一个空的DataFrame来存储所有数据
    three_topFeatureshapdata = pd.DataFrame()
    topFeatureshapdata= pd.DataFrame()
    Data_DIR = "D:/shap each sample/"
    sampleNum = 34840
    # 循环读取每个CSV文件，并将它们追加到all_data DataFrame中
    for i in range(sampleNum):  #  sampleNum
        sampleShapi = pd.read_csv(Data_DIR + str(i) + ".csv", compression='gzip', header=None)
        if sampleShapi.iloc[:, 0].max() != 1.0:
            print(sampleNum)
        sampleShapiTopFeature = pd.DataFrame(sampleShapi.iloc[topFeaturesShapIndices,0])
        three_topFeatureshapdata = pd.concat([three_topFeatureshapdata, sampleShapiTopFeature], ignore_index=True, axis=1)

    three_topFeatureshapNParray = three_topFeatureshapdata.to_numpy().T

    labelArray = np.array(['out layer top score', 'mid layer top score',  'inner score'])  # 做图的label

    # 设置图形和GridSpec布局
    fig = plt.figure()
    gs = GridSpec(1, 5, width_ratios=[0.01, 1, 1, 1, 0.05], wspace=0.2, hspace=0.2)

    # 行标签
    row_labels = ['SHAP']

    # 一行 'SHAP'
    # 添加行标签（最左侧）
    ax_label = fig.add_subplot(gs[0, 0])
    ax_label.text(0.5, 0.5, row_labels[0],
                  rotation=90, va='center', ha='center', fontsize=14)
    ax_label.axis('off')  # 隐藏坐标轴
    # 创建3个子图
    for j in range(3):
        topFeaturesShapdata = three_topFeatureshapNParray[:, j].reshape(134, 260).copy()
        topFeaturesShapdata = topFeaturesShapdata / np.max(topFeaturesShapdata)
        print(topFeaturesShapIndices[j])
        print(np.max(topFeaturesShapdata))
        ax = fig.add_subplot(gs[0, j + 1])  # 使用中间3列
        img = ax.imshow(topFeaturesShapdata, cmap='viridis', aspect='auto')
        ax.set_title(labelArray[j]+':\n'+"m/z value "+str(np.array([502.975, 741.541, 798.54])[j]))

        # 隐藏右侧和顶部边框
        ax.spines[['right', 'top']].set_visible(False)
        ax.axis('off')  # 隐藏坐标轴

    # 添加行colorbar（最右侧）
    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(img, cax=cax, label='SHAP value')

    # 设置总标题
    # plt.suptitle('Subplots with Row Labels and Color Bars', fontsize=16, y=0.95)

    plt.show()