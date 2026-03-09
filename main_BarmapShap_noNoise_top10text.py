import numpy as np
import matplotlib.pyplot as plt
import datatable as dt

if __name__ == "__main__":
    file = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/shap/sum_shap_heatmap_nonorm check_additivity.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datay = my_table.to_numpy()

    file = "MSI/data/mz list.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datax = my_table.to_numpy()

    # 归一化
    datay[:, 0] = datay[:, 0] / np.max(datay[:, 0])
    datay[:, 1] = datay[:, 1] / np.max(datay[:, 1])
    datay[:, 2] = datay[:, 2] / np.max(datay[:, 2])
    noiseThreshold = 0.01

    # Class 1
    #
    #
    plt.subplot(3, 1, 1)
    plt.ylabel('out layer score')

    # 筛出数值超过0.05的数据，也就是重要性低的数据都筛掉
    dataNotNoiseIndices = np.where(datay[:, 0] > noiseThreshold)[0]
    print(dataNotNoiseIndices.shape)
    # 筛出数据做bar图
    plt.bar(datax[dataNotNoiseIndices, 0], datay[dataNotNoiseIndices, 0], width=1, bottom=None, align='center', )

    # 获取排序后的索引
    sorted_indices = np.argsort(datay[:, 0])[::-1]  # 升序排序后取反得到降序索引
    # 获取前10个最大元素的索引
    top10_indices = sorted_indices[:10]
    top_10_values = [datay[:, 0][i] for i in top10_indices]
    top_10_categories = [datax[i, 0] for i in top10_indices]

    # 标注最高的10个条形
    for i, val in enumerate(top_10_values):
        plt.text(top_10_categories[i],
                 (10-i) / 10,
                 str(top_10_categories[i]) + '--' + str(np.round(top_10_values[i], 3)),
                 ha='center',
                 va='center')  # 在每个条形中部添加文本标签
    plt.xlim(400, 1000)

    # Class 2
     #
    #
    plt.subplot(3, 1, 2)
    plt.ylabel('mid layer score')

    # 筛出数值超过0.05的数据，也就是重要性低的数据都筛掉
    dataNotNoiseIndices = np.where(datay[:, 1] > noiseThreshold)[0]
    print(dataNotNoiseIndices.shape)
    # 筛出数据做bar图
    plt.bar(datax[dataNotNoiseIndices, 0], datay[dataNotNoiseIndices, 1], width=1, bottom=None, align='center', )

    # 获取排序后的索引
    sorted_indices = np.argsort(datay[:, 1])[::-1]  # 升序排序后取反得到降序索引
    # 获取前10个最大元素的索引
    top10_indices = sorted_indices[:10]
    top_10_values = [datay[:, 1][i] for i in top10_indices]
    top_10_categories = [datax[i, 0] for i in top10_indices]

    # 标注最高的10个条形
    for i, val in enumerate(top_10_values):
        plt.text(top_10_categories[i],
                 (10-i) / 10,
                 str(top_10_categories[i]) + '--' + str(np.round(top_10_values[i], 3)),
                 ha='center',
                 va='center')  # 在每个条形中部添加文本标签
    plt.xlim(400, 1000)

    # Class 3
    plt.subplot(3, 1, 3)
    plt.ylabel('inner score')
    plt.xlabel('m/z value')

    # 筛出数值超过0.05的数据，也就是重要性低的数据都筛掉
    dataNotNoiseIndices = np.where(datay[:, 2] > noiseThreshold)[0]
    print(dataNotNoiseIndices.shape)
    # 筛出数据做bar图
    plt.bar(datax[dataNotNoiseIndices, 0], datay[dataNotNoiseIndices, 2], width=1, bottom=None, align='center', )

    # 获取排序后的索引
    sorted_indices = np.argsort(datay[:, 2])[::-1]  # 升序排序后取反得到降序索引
    # 获取前10个最大元素的索引
    top10_indices = sorted_indices[:10]
    top_10_values = [datay[:, 2][i] for i in top10_indices]
    top_10_categories = [datax[i, 0] for i in top10_indices]

    # 标注最高的10个条形
    for i, val in enumerate(top_10_values):
        plt.text(top_10_categories[i],
                 (10-i) / 10,
                 str(top_10_categories[i]) + '--' + str(np.round(top_10_values[i], 3)),
                 ha='center',
                 va='center')  # 在每个条形中部添加文本标签
    plt.xlim(400, 1000)
    plt.show()