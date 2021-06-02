'''
Author: your name
Date: 2021-04-21 10:43:29
LastEditTime: 2021-05-18 15:41:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \code\handwriting_recognition_simple.py
'''
import os
import datetime
import numpy as np
import tensorflow as tf
from data_loader import MNISTLoader

'''
    Keras Sequential API 方式创建简单网络模型；
    通过向tf.kears.models.Sequential()提供一个层的列表，快速建立一个tf.kears.Model模型
'''

# CNN模型
cnn_modle = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = [5,5], padding = 'same', activation= tf.nn.relu), # 第1卷积层
    tf.keras.layers.MaxPool2D( pool_size = [2, 2],strides= 2), # 池化层
    tf.keras.layers.Conv2D(filters = 64, kernel_size = [5,5], padding = 'same', activation= tf.nn.relu), # 第2卷积层
    tf.keras.layers.MaxPool2D( pool_size = [2, 2],strides= 2), # 池化层
    tf.keras.layers.Flatten(), # 将图片拉直为一维向量
    tf.keras.layers.Dense(units = 100, activation = tf.nn.relu), # 隐含层：100个神经元，激活函数ReLU
    tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax) # 输出层：10个神经元，激活函数softmax
])

'''
    当模型建立完成后，通过tf.kears.Model的complie方法，配置训练过程
'''

# 模型配置
cnn_modle.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), # 优化器
    loss = tf.keras.losses.sparse_categorical_crossentropy, # 损失函数
    metrics = [tf.keras.metrics.sparse_categorical_accuracy] # 评估指标 
)

'''
    当模型配置完成后，通过tf.kears.Model的fit方法训练模型
'''

# 实例化数据对象
dataLoader = MNISTLoader()

# 创建TensorBoard回调函数
fit_log_dir = './tensorboard/bySimple/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = fit_log_dir, # 输出路径
    histogram_freq = 1, # 统计每层直方图
    profile_batch = 0, # 不启动profile
    update_freq = 'batch' # 更新频次，以batch
)
os.makedirs(fit_log_dir + '/train/') # 创建目录

cnn_modle.fit(
    x = dataLoader.train_data, # 训练数据
    y = dataLoader.train_label, # 标签数据
    validation_split=0.3, # 验证集划分
    batch_size = 50, # 批次大小
    epochs = 5, # 将训练数据迭代多少次
    callbacks = [tensorboard_callback] # tensorboard回调函数
)

'''
    当模型训练完成之后，保存模型
'''
save_path = './save/bySimple/'
tf.saved_model.save(cnn_modle, save_path)


'''
    当模型训练完成之后，通过tf.kears.Model的evaluate评估训练效果
'''


# test accuracy: [0.04578051716089249, 0.9861999750137329]
print('test accuracy: {0}'.format(cnn_modle.evaluate(
    dataLoader.test_data,
    dataLoader.test_label   
)))





