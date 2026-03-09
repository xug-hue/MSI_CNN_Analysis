import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datatable as dt

if __name__ == "__main__":
    Data_DIR = "MSI/data/MSINorm-all.csv"
    my_table = dt.fread(Data_DIR, sep=",", header=False)  ## datatable格式读取文件
    data = my_table.to_numpy()  ## datatable格式转np数组，保存int数据
    print(data.shape)

    file = "MSI/data/mz list.csv"
    my_table = dt.fread(file, sep=",", header=False)  ## datatable格式读取文件
    datax = my_table.to_numpy()

    Result_DIR = "MSI/result/HR2MSI/cnn model/hc3tsne2 best/intensityHeatmap"

    topFeaturesGradcamIndices = np.array([1, 1, 1])
    topFeaturesShapIndices = np.array([1, 1, 1])
    topFeaturesGradcamIndices[0] = np.where(datax == np.array([633.034]))[0]
    topFeaturesGradcamIndices[1] = np.where(datax == np.array([502.975]))[0]
    topFeaturesGradcamIndices[2] = np.where(datax == np.array([798.556]))[0]
    topFeaturesShapIndices[0] = np.where(datax == np.array([502.975]))[0]
    topFeaturesShapIndices[1] = np.where(datax == np.array([741.541]))[0]
    topFeaturesShapIndices[2] = np.where(datax == np.array([798.54]))[0]


    for i in range(len(topFeaturesGradcamIndices)):
        topFeaturesGradcamdata = data[:, topFeaturesGradcamIndices[i]].reshape(134, 260).copy()
        topFeaturesGradcamdata = topFeaturesGradcamdata / np.max(topFeaturesGradcamdata)

        fig, ax = plt.subplots(figsize=(8, 6))
        # 创建热力图
        im = ax.imshow(topFeaturesGradcamdata, cmap='viridis', aspect='auto')
        # 创建等高的颜色条轴
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 关键参数
        # 添加颜色条
        cbar = plt.colorbar(im, cax=cax)
        # 优化布局
        plt.tight_layout()
        # 保存图片
        plt.savefig(Result_DIR + str(np.array([633.034, 502.975, 798.556])[i]) + "Gradcam.jpg",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=-0.01)

    for i in range(len(topFeaturesShapIndices)):
        topFeaturesShapdata = data[:, topFeaturesShapIndices[i]].reshape(134, 260)
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
        plt.savefig(Result_DIR + str(np.array([502.975, 741.541, 798.54])[i]) + "SHAP.jpg",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=-0.01)
    # 显示图形
    plt.show()