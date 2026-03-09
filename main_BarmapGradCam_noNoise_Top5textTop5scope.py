import numpy as np
import matplotlib.pyplot as plt
import datatable as dt

if __name__ == "__main__":
    file = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/Grad-CAM/sum_Gradcam_peakHeatmap_nonorm second time.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datay = my_table.to_numpy()

    file = "MSI/data/mz list.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datax = my_table.to_numpy()

    # 模型视野决定每个热力值的作用范围，本案例是400到1000的某一段范围
    modelScope = dt.fread('MSI/data/cnnModelScope.csv', sep=",", header=False).to_numpy()

    # 读取总热力图_没有正则化数据，本案例是180个维度
    total_heatmap_nonorm = dt.fread('MSI/result/HR2MSI/cnn model/hc3tsne2 best/Grad-CAM/sum_heatmap_nonorm.csv',
                                    sep=",", header=False).to_numpy()

    classNum = 3  # 分类数
    top_scopes_num = 10  # 标注最大视野数
    top_data_num = 1  # 标注视野最大数据数
    noiseThreshold = 0.01

    labelArray = np.array(['out layer score', 'mid layer score',  'inner score'])  # 做图的label
    # 循环处理每个类并作图
    for classID in range(classNum):
        # 归一化
        datay[:, classID] = datay[:, classID] / np.max(datay[:, classID])
        plt.subplot(classNum, 1, classID+1)
        plt.ylabel(labelArray[classID])

        # 筛出数值超过阈值的数据，也就是重要性低的数据都筛掉
        dataNotNoiseIndices = np.where(datay[:, classID] > noiseThreshold)[0]
        print(dataNotNoiseIndices.shape)
        # 筛出数据做bar图
        plt.bar(datax[dataNotNoiseIndices, 0], datay[dataNotNoiseIndices, classID],
                width=1, bottom=None, align='center')
        plt.xlim(400, 1000)

        # 升序排序后取后几个索引，将modelScope对应索引的范围读出
        top_heat_scopes = modelScope[np.argsort(total_heatmap_nonorm[:, classID])[-top_scopes_num:], :]
        print(top_heat_scopes)

        top_indices_all = np.array([-1])  # 设置所有标注索引，防止重复
        # 为几个最热视野中的每一个标注最重要的5个数据坐标到bar图
        for scopei in range(len(top_heat_scopes)-1, -1, -1):
            # 读出topn的第i个视野中的datay数据片段
            datay_segment = datay[top_heat_scopes[scopei, 0]:top_heat_scopes[scopei, 1], classID]

            # 获取datay数据片段排序后的索引，找出视野内最大的几个数据点索引，此时是相对的索引
            sorted_indices = np.argsort(datay_segment)[::-1]  # 升序排序后取反得到降序索引
            top_indices = sorted_indices[:top_data_num]

            # 计算在原始数组中的索引（需要加上起始索引），即绝对索引
            top_indices_in_original_array = top_indices + top_heat_scopes[scopei, 0]

            # 与所有标注索引重复的删除，只标注不重复的
            top_indices_in_original_array = \
                top_indices_in_original_array[~np.isin(top_indices_in_original_array, top_indices_all)]
            top_indices_all = np.append(top_indices_all, top_indices_in_original_array)

            # 获取前5个最大元素横纵坐标
            top_values = [datay[:, classID][i] for i in top_indices_in_original_array]
            top_categories = [datax[i, 0] for i in top_indices_in_original_array]

            # 标注该视野内前5个数据的坐标到bar图
            for i, val in enumerate(top_values):
                plt.text(top_categories[i],
                         scopei/top_scopes_num+(top_data_num-i) / top_data_num / top_scopes_num,  # text高度，尝试不重叠
                         str(np.round(top_categories[i], 3)) + '--' + str(np.round(top_values[i], 3)),
                         ha='center',
                         va='center',
                         fontsize=10)  # 字体太大会是标注重叠


            if scopei == len(top_heat_scopes)-1:
                plt.xlabel('m/z value')

    plt.show()