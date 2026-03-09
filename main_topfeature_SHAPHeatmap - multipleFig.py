import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datatable as dt

if __name__ == "__main__":
    Result_DIR = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/ShapHeatmap"

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
        sampleShapiTopFeature = pd.DataFrame(sampleShapi.iloc[topFeaturesShapIndices, 0])
        three_topFeatureshapdata = pd.concat([three_topFeatureshapdata, sampleShapiTopFeature], ignore_index=True, axis=1)

    three_topFeatureshapNParray = three_topFeatureshapdata.to_numpy().T

    for i in range(len(topFeaturesShapIndices)):
        topFeaturesShapdata = three_topFeatureshapNParray[:, i].reshape(134, 260).copy()
        topFeaturesShapdata = topFeaturesShapdata / np.max(topFeaturesShapdata)

        fig, ax = plt.subplots(figsize=(8, 6))
        # 创建热力图
        im = ax.imshow(topFeaturesShapdata, cmap='viridis', aspect='auto')
        # 创建等高的颜色条轴
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 关键参数
        # 添加颜色条
        cbar = plt.colorbar(im, cax=cax)
        # 优化布局
        plt.tight_layout()
        # 保存图片
        plt.savefig(Result_DIR + str(np.array([502.975, 741.541, 798.54])[i]) + ".jpg",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=-0.01)
    # 显示图形
    plt.show()
