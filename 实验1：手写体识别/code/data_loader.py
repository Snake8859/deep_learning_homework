'''
Author: snake8859
Date: 2021-04-21 10:43:45
LastEditTime: 2021-05-18 14:35:50
LastEditors: Please set LastEditors
Description: 手写体数据集加载器
FilePath: \code\data_loader.py
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class MNISTLoader():
    '''MNIST 数据获取和预处理'''
    def __init__(self):
        # 自动下载MNIST数据集
        mnist = tf.keras.datasets.mnist
        # 训练集和测试集划分
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()

        # 训练集和测试集增加一个颜色通道维度，其值为像素值(unint8)归一化到0-1(float32)
        '''
            在 TensorFlow 中，图像数据集的一种典型表示是 [图像数目，长，宽，色彩通道数] 的四维张量。
        '''
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis = -1) # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis = -1) # [10000, 28, 28, 1]

        self.train_label = self.train_label.astype(np.int32) # [60000]
        self.test_label = self.test_label.astype(np.int32) # [10000]

        # 计算训练集和测试集的个数
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        '''
        @description: 随机从数据集取出batch_size个数据对象并返回
        @param 
            batch_size 每批数据对象个数
        @return 
            train_data  数据
            train_label 标签
        '''
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


    def show_img(self, batch_size = 16):
        '''
        @description: 显示数据和标签
        @param 
            batch_size 显示数据个数, 默认16
        @return None
        '''
        show_data, show_label = self.get_batch(batch_size)
        # print(show_data.shape, show_data.dtype, type(show_data)) # (batch_size, 28, 28, 1)
        col = 0
        for i in range(batch_size):
            img = show_data[i]
            img_label = show_label[i]
            # print(img.shape, img_label.shape) # (28, 28, 1)
            cur_spec = (col, i % 4)
            if (i + 1) % 4 == 0: # 每4个换行
                col +=1
            plt.subplot2grid((4, 4), cur_spec)
            plt.imshow(img, cmap='gray')
            plt.title(img_label)
            plt.axis('off')

        plt.show()


if __name__ == "__main__":
    data_loader = MNISTLoader()
    data_loader.show_img(16)