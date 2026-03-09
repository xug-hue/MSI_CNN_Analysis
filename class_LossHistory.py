from tensorflow import keras
import matplotlib.pyplot as plt
import csv


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
    # 按照batch来进⾏追加数据

    def on_batch_end(self, batch, logs={}):
        # 每⼀个batch完成后向容器⾥⾯追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        '''
        #每五秒按照当前容器⾥的值来绘图
        if int(time.time()) % 10 == 0:
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
                    '''

    def on_epoch_end(self, batch, logs={}):
        # 每⼀个epoch完成后向容器⾥⾯追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        '''
        # 每五秒按照当前容器⾥的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
        '''

    # 绘图，这⾥把每⼀种曲线都单独绘图，若想把各种曲线绘制在⼀张图上的话可修改此⽅法
    def draw_epoch(self, lists, label, type):
        plt.figure()
        #plt.ylim(0.0, 1.0)
        plt.plot(range(len(lists)), lists, 'r', marker='.', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig("MSI/result/HR2MSI/cnn model/"+type+'_'+label+'.jpg')
        plt.close()

    def draw_batch(self, lists, label, type):
        plt.figure()
        #plt.ylim(0.0, 1.0)
        plt.plot(range(len(lists)), lists, 'r', marker='.',markevery=200, label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig("MSI/result/HR2MSI/cnn model/"+type+'_'+label+'.jpg')
        plt.close()

    def draw_batch_lim(self, lists, label, type):
        plt.figure()
        plt.xlim(0, 310)
        plt.plot(range(len(lists)), lists, 'r', marker='.', markevery=10, label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig("MSI/result/HR2MSI/cnn model/"+type+'_lim_'+label+'.jpg')
        plt.close()

    # 由于这⾥的绘图设置的是5s绘制⼀次，当训练结束后得到的图可能不是⼀个完整的训练过程（最后⼀次绘图结束，有训练了0-5秒的时间）
    # 所以这⾥的⽅法会在整个训练结束以后调⽤
    def end_draw(self):
        self.draw_batch(self.losses['batch'], 'loss', 'train_batch')
        self.draw_batch_lim(self.losses['batch'], 'loss', 'train_batch')
        self.draw_batch(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_batch_lim(self.accuracy['batch'], 'acc', 'train_batch')
        #self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_epoch(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_epoch(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_epoch(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_epoch(self.val_acc['epoch'], 'acc', 'val_epoch')

    def end_save(self, file):
        file = "MSI/result/HR2MSI/cnn model/epoch.csv"
        with open(file, 'w', encoding='utf-8', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['acc epoch', 'val_acc epoch'])
            for j in range(len(self.accuracy)):
                csv_write = csv.writer(f)
                csv_write.writerow([self.accuracy['epoch'][j], self.val_acc['epoch'][j]])

        file = "MSI/result/HR2MSI/cnn model/batch.csv"
        with open(file, 'w', encoding='utf-8', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['acc batch', 'val_acc batch'])
            for j in range(len(self.accuracy)):
                csv_write = csv.writer(f)
                csv_write.writerow([self.accuracy['batch'][j], self.val_acc['batch'][j]])
