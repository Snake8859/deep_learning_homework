'''
Author: your name
Date: 2021-06-12 19:39:53
LastEditTime: 2021-06-12 22:23:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \code\exam_test.py
'''
import numpy as np
from numpy.core.numeric import outer
import tensorflow as tf
import tensorflow_addons as tfa


def test1():
    input_data = np.array([[5,6,0,1,8,2],[2,5,7,2,3,7],[0,7,2,4,5,6],[5,3,6,9,3,1],[6,5,3,1,4,6],[5,2,4,0,8,7]])
    input_data = tf.convert_to_tensor(input_data)
    input_data = tf.reshape(input_data,shape=[1,6,6,1])
    # print(input_data)
    filter_1 = np.array([[1,-1,0],[-1,1,-1],[0,-1,1]])
    filter_1 = tf.convert_to_tensor(filter_1)
    filter_1 = tf.reshape(filter_1, shape=[3,3,1,1])

    filter_2 = np.array([[-1,2,-1],[1,5,1],[-1,0,-1]])
    filter_2 = tf.convert_to_tensor(filter_2)
    filter_2 = tf.reshape(filter_2, shape=[3,3,1,1])

    # print(filter_1)
    # print(filter_2)

    # ！！！用维数拼接而不是reshape
    
    my_filter = tf.concat([filter_1,filter_2], 3)

    # print(my_filter)
    # exit()

    out_same1 = tf.nn.conv2d(input_data, filter_1,[1,1,1,1], padding='VALID') # [1,4,4,1]

    out_same1 = tf.nn.relu(out_same1)
    print(out_same1.numpy().reshape(4,4))

    out_same2 = tf.nn.conv2d(input_data, filter_2, [1,1,1,1], padding='VALID')
    out_same2 = tf.nn.relu(out_same2)
    print(out_same2.numpy().reshape(4,4))

    print('=' * 100)

    out_same3 = tf.nn.conv2d(input_data, my_filter, [1,1,1,1], padding='VALID') # [1,4,4,2]
    out_same3 = tf.nn.relu(out_same3)
    print(out_same3.numpy())


def test2():
    gt = np.array([[0,1,0,0,1]],dtype=np.float32)
    pred = np.array([[0.2,0.8,0.4,0.1,0.9]], dtype=np.float32)

    ce_loss = tf.losses.binary_crossentropy(gt, pred)
    fl = tfa.losses.SigmoidFocalCrossEntropy(alpha = 0.4, gamma = 2)
    f_loss = fl(gt, pred)
    print('交叉熵损失:{0}'.format(ce_loss))
    print('焦点损失：{0}'.format(f_loss))


test1()
# test2()