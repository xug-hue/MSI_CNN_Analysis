import class_LossHistory
import numpy as np
import pandas as pd
import datatable as dt
import math
import os
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


BASE_DIR = "MSI/data/HR2MSINorm/"
CSV_DIR = "MSI/data/MSIbiaozhuNPArray_3class_clusteringHC3tSNE2.csv"
MANIFEST_DIR = "MSI/data/MSIbiaozhuShuffle.csv"

pixelNumX = 260
sample_num = 0
Batch_size = 200
mz_num = 92064
#打开CSV数据文件，
def get_feature(csv_file_num,BASE_DIR=BASE_DIR):
    #csv_file =str(Path(__file__).parent / 'MSI/result/HR2MSI/lxml0(0,0).csv')
    file = "norm-lxml"+str(csv_file_num)+"("+ str(csv_file_num // pixelNumX) \
                   + ',' + str(csv_file_num % pixelNumX) + ')' + '.csv'
    my_table = dt.fread(BASE_DIR + file, sep=",", header=False)  ## datatable格式读取文件
    ints = my_table.to_numpy()      ## datatable格式转np数组，保存int数据
    if ints.shape != (92064, 1):
        print(ints.shape, csv_file_num)
    return ints

#把标签转成oneHot形式
def convert2oneHot(index,classNum):
    hot = np.zeros((classNum,))
    hot[index-1] = 1
    return hot

#标签处理方式

def create_2label_csv(CSV_DIR=CSV_DIR):
    lists = pd.read_csv(CSV_DIR, sep=r",", header=None)
    lists[lists[0].isin([3])] = 2   #把标签3改为2
    lists[lists[0].isin([5])] = 2   #把标签5改为2
    lists = lists[lists[0].isin([1, 2])]     #提取1，2标签重做list
    lists = lists.sample(frac=1)             #随机打乱list中的样本
    lists.to_csv(MANIFEST_DIR, index=1, header=None)  #保存打乱后的CSV文件，所有选取样本的原始行索引会输出，index=1
    print(lists.value_counts())      #打印所有标签的样本数
    print("Finish save csv")

def create_3label_csv(CSV_DIR=CSV_DIR):
    lists = pd.read_csv(CSV_DIR, sep=r",", header=None)
    lists = lists[lists[0].isin([1, 2, 3])]     #提取1，2，3标签重做list
    lists = lists.sample(frac=1)             #随机打乱list中的样本
    lists.to_csv(MANIFEST_DIR, index=1, header=None)  #保存打乱后的CSV文件，所有选取样本的原始行索引会输出，index=1
    print(lists.value_counts())      #打印所有标签的样本数
    print("Finish save csv")

#数据迭代方式

def xs_gen(path=MANIFEST_DIR,batch_size = Batch_size,train=True):

    img_list = pd.read_csv(path)
    img_list = np.array(img_list)
    #np.random.shuffle(img_list)
    print("data",sample_num)
    if train:
        img_list = img_list[:math.ceil(sample_num/10*9)]
        print("Found %s train items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = img_list[math.ceil(sample_num/10*9):]
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([get_feature(fileNum) for fileNum in batch_list[:,0]])
            batch_y = np.array([convert2oneHot(label, 3) for label in batch_list[:,1]])
            # 3个类的标签
            yield batch_x, batch_y

#模型搭建
def build_model(input_shape=(mz_num, 1),num_classes=3):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=16,strides=2, activation='relu',input_shape=input_shape))
    model.add(Conv1D(filters=16, kernel_size=16,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=64, kernel_size=8,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(filters=64, kernel_size=8,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=128, kernel_size=4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(filters=128, kernel_size=4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=256, kernel_size=2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(filters=256, kernel_size=2,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)

#模型训练
if __name__ == "__main__":
    """dat1 = get_feature("TRAIN101.mat")
    print("one data shape is",dat1.shape)
    #one data shape is (12, 5000)
    plt.plot(dat1[0])
    plt.show()"""
    if not os.path.exists(MANIFEST_DIR):
        create_3label_csv()
    img_list = pd.read_csv(MANIFEST_DIR)
    sample_num = len(img_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_iter = xs_gen(train=True)
    test_iter = xs_gen(train=False)
    model = build_model()
    print(model.summary())

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='msi_1dcnn_best_model.{epoch:02d}-{val_acc:.2f}.h5',
        monitor='val_acc', save_best_only=True, verbose=1)

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=0,
        embeddings_freq=0,
        #update_freq='batch'
    )

    logs_loss = class_LossHistory.LossHistory()

    model.fit_generator(
        generator=train_iter,
        steps_per_epoch=math.ceil(sample_num/10*9)//Batch_size,
        epochs=50,
        initial_epoch=0,
        validation_data=test_iter,
        # keras 2之下的版本 nb_val_samples = 100//Batch_size,
        validation_steps=(sample_num-math.ceil(sample_num/10*9))//Batch_size,
        callbacks=[ckpt, tensorboard_cb, logs_loss],
        )

    #作图

    logs_loss.end_draw()

