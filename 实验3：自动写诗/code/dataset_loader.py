'''
Author: snake8859
Date: 2021-06-06 11:30:16
LastEditTime: 2021-06-13 13:27:22
LastEditors: Please set LastEditors
Description: 
    古诗数据生成器。参考：https://blog.csdn.net/aaronjny/article/details/103806954
FilePath: \code\poetryDataGenerator.py
'''
import math
import settings
import numpy as np
import tensorflow as tf


class PoetryDataGenerator:
    """
    古诗数据集生成器
    """

    def __init__(self, data, random=False):
        # 数据集
        self.data = data
        # batch size
        self.batch_size = settings.BATCH_SIZE
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))
        # 每个epoch开始时是否随机混洗
        self.random = random

    def __len__(self):
        return self.steps


    def sequence_padding(self, data, length=None, padding=None):
        """
        将给定数据填充到相同长度
        :param data: 待填充数据
        :param length: 填充后的长度，不传递此参数则使用data中的最大长度
        :param padding: 用于填充的数据，不传递此参数则使用[PAD]的对应编号
        :return: 填充后的数据
        """
        # 计算填充长度
        if length is None:
            length = max(map(len, data)) # batch_size 最长的长度
        # 计算填充数据
        if padding is None:
            padding = 8292
        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line)
            # 不足就进行填充
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            # 超过就进行截断
            else:
                outputs.append(line[:length])
        return np.array(outputs)

    def __iter__(self):
        total = len(self.data)
        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_data = self.data[start:end]
            # 填充为相同长度
            batch_data = self.sequence_padding(batch_data)
            # print(batch_data, batch_data.shape)
            # yield x,y
            '''
                将诗的内容错开一位分别作为数据和标签
                    example:
                    输入: [START] 床 前 明 月 光 ， 疑 是 地 上 霜 。 举 头 望 明 月 ， 低 头 思 故 乡 。
                    输出: 床    前 明 月 光 ， 疑 是 地 上 霜 。 举 头 望 明 月 ， 低 头 思 故 乡 。 [EOP]
                还有一点不同的是，标签部分使用了one-hot进行处理，而数据部分没有使用。
                原因在于，数据部分准备输入词嵌入层，而词嵌入层的输入不需要进行one-hot；而标签部分，需要和模型的输出计算交叉熵，输出层的激活函数是softmax，所以标签部分也要转成相应的shape，故使用one-hot形式。
             '''
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], settings.VOCAB_SIZE)
            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
            写成生成器的形式，主要出于内存方面的考虑。
            训练时需要对数据进行填充、转one-hot形式等操作，会占用较多内存。
            如果提前对全部数据都进行处理，内存可能会溢出。而以生成器的形式，可以只在要进行训练的时候，处理相应batch size的数据即可。
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()