import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datatable as dt

if __name__ == "__main__":
    Data_DIR = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/Grad-CAM/peakHeatmapInEachSample.csv"
    my_table = dt.fread(Data_DIR, sep=",", header=False)  ## datatable格式读取文件
    data = my_table.to_numpy().T  ## datatable格式转np数组，保存int数据
    print(data.shape)
    print(np.max(data[0, :]))
    print(np.max(data[10, :]))
    print(np.max(data[10000, :]))
    print(np.max(data[26010, :]))

    file = "MSI/data/mz list.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datax = my_table.to_numpy()

    topFeaturesGradcamIndices = np.array([1, 1, 1])
    topFeaturesGradcamIndices[0] = np.where(datax == np.array([633.034]))[0]
    topFeaturesGradcamIndices[1] = np.where(datax == np.array([502.975]))[0]
    topFeaturesGradcamIndices[2] = np.where(datax == np.array([798.556]))[0]


    labelArray = np.array(['out layer top score', 'mid layer top score',  'inner score'])  # 做图的label

    # 设置图形和GridSpec布局
    fig = plt.figure()
    gs = GridSpec(1, 5, width_ratios=[0.01, 1, 1, 1, 0.05], wspace=0.2, hspace=0.2)

    # 行标签
    row_labels = ['Grad-CAM']

    # 第一行'Grad-CAM'
    # 添加行标签（最左侧）
    ax_label = fig.add_subplot(gs[0, 0])
    ax_label.text(0.5, 0.5, row_labels[0],
                  rotation=90, va='center', ha='center', fontsize=14)
    ax_label.axis('off')  # 隐藏坐标轴
    # 创建3个子图
    for j in range(3):
        topFeaturesGradcamdata = np.nan_to_num(data[:, topFeaturesGradcamIndices[j]].reshape(134, 260)).copy()
        topFeaturesGradcamdata = topFeaturesGradcamdata / np.max(topFeaturesGradcamdata)
        print(topFeaturesGradcamIndices[j])
        print(np.max(topFeaturesGradcamdata))

        ax = fig.add_subplot(gs[0, j + 1])  # 使用中间3列
        img = ax.imshow(topFeaturesGradcamdata, cmap='viridis', aspect='auto')
        ax.set_title(labelArray[j] + ":\n" + "m/z value:" + str(np.array([633.034, 502.975, 798.556])[j]))

        # 隐藏右侧和顶部边框
        ax.spines[['right', 'top']].set_visible(False)
        ax.axis('off')  # 隐藏坐标轴

    # 添加行colorbar（最右侧）
    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(img, cax=cax, label='Grad_CAM value')

    # 设置总标题
    # plt.suptitle('Subplots with Row Labels and Color Bars', fontsize=16, y=0.95)

    plt.show()