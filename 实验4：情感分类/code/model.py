'''
Author: snake8859
Date: 2021-06-19 17:15:50
LastEditTime: 2021-06-22 11:26:09
LastEditors: Please set LastEditors
Description: 构建TextCNN模型
FilePath: \code\model.py
'''

from tensorflow.python.keras.backend import dtype
import settings
import tensorflow as tf
from tensorflow import keras
from data_process import *

def TextCNN_Old(max_sen_len, embedding_dim, filters_size, num_filters, dropout_rate, num_classes, regularizers_lambda):
    '''
    @description: TextCNN模型(废弃)
    @param 
        max_sen_len 句子长度
        embedding_dim 词向量维度
        filters_size 卷积核长度
        num_filters 卷积核个数
        dropout_rate 丢弃率
        num_classes 分类个数
        regularizers_lambda 正则化率
    @return model 模型
    '''    
    # 输入词向量（预训练词向量）
    inputs = keras.Input(shape = (max_sen_len, embedding_dim))
    # 增加通道数
    feature_map = keras.layers.Reshape((max_sen_len, embedding_dim, 1), name='add_channel')(inputs)

    pool_outputs = []

    # 定义不同大小的卷积核(3,4,5)，提取局部文本特征
    for filter_size in list(map(int, filters_size.split(','))):
        filter_shape = (filter_size, embedding_dim)

        conv = keras.layers.Conv2D( num_filters, 
                                    filter_shape, strides=(1,1), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    activation='relu', 
                                    name = 'convolution_{:d}'.format(filter_size))(feature_map)

        # max-over-time pooling
        max_pool_shape = (max_sen_len - filter_size + 1, 1)
        # print(max_pool_shape)
        pool = keras.layers.MaxPool2D(  pool_size= max_pool_shape, 
                                        strides=(1,1), 
                                        padding='valid', 
                                        data_format='channels_last', 
                                        name ='max_pooling_{:d}'.format(filter_size))(conv)

        pool_outputs.append(pool)
    # 拼接
    pool_outputs = keras.layers.concatenate(pool_outputs, axis = -1, name = 'concatenate') 
    # 展开
    pool_outputs = keras.layers.Flatten(data_format= 'channels_last', name = 'flatten')(pool_outputs)
    pool_outputs = keras.layers.Dropout(dropout_rate, name = 'dropout')(pool_outputs)

    outputs = keras.layers.Dense( num_classes, 
                                  activation='softmax',
                                  kernel_regularizer= keras.regularizers.l2(regularizers_lambda),
                                  bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                  name = 'dense')(pool_outputs)
    
    model = keras.Model(inputs = inputs, outputs = outputs)

    return model


def TextCNN(update_w2v, max_sen_len, embedding_matrix, embedding_dim, filters_size, num_filters, dropout_rate, num_classes, regularizers_lambda):
    '''
    @description: TextCNN模型
    @param 
        update_w2v 是否更新预训练词向量
        max_sen_len 句子长度
        embedding_matrix 预训练词向量
        embedding_dim 词向量维度
        filters_size 卷积核长度
        num_filters 卷积核个数
        dropout_rate 丢弃率
        num_classes 分类个数
        regularizers_lambda 正则化率
    @return model 模型
    '''    
    # 输入词向量（预训练词向量）
    inputs = keras.Input(shape = (max_sen_len, ), dtype = tf.uint16)

    # 词嵌入层
    embed = keras.layers.Embedding(
                                    settings.vocab_size, 
                                    settings.embedding_dim, 
                                    input_length= max_sen_len,
                                    embeddings_initializer = keras.initializers.Constant(embedding_matrix),
                                    trainable=update_w2v,
                                    name='embedding')(inputs)

    # 增加通道数
    feature_map = keras.layers.Reshape((max_sen_len, embedding_dim, 1), name='add_channel')(embed)

    pool_outputs = []

    # 定义不同大小的卷积核(3,4,5)，提取局部文本特征
    for filter_size in list(map(int, filters_size.split(','))):
        filter_shape = (filter_size, embedding_dim)

        conv = keras.layers.Conv2D( num_filters, 
                                    filter_shape, strides=(1,1), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    activation='relu', 
                                    name = 'convolution_{:d}'.format(filter_size))(feature_map)

        # max-over-time pooling
        max_pool_shape = (max_sen_len - filter_size + 1, 1)
        # print(max_pool_shape)
        pool = keras.layers.MaxPool2D(  pool_size= max_pool_shape, 
                                        strides=(1,1), 
                                        padding='valid', 
                                        data_format='channels_last', 
                                        name ='max_pooling_{:d}'.format(filter_size))(conv)

        pool_outputs.append(pool)
    # 拼接
    pool_outputs = keras.layers.concatenate(pool_outputs, axis = -1, name = 'concatenate') 
    # 展开
    pool_outputs = keras.layers.Flatten(data_format= 'channels_last', name = 'flatten')(pool_outputs)
    pool_outputs = keras.layers.Dropout(dropout_rate, name = 'dropout1')(pool_outputs) # 丢弃
    
    # if(update_w2v):
    #     # 全连接层1
    #     dense1 = keras.layers.Dense(units= 128, activation='relu', name = 'dense1')(pool_outputs)
    #     pool_outputs = keras.layers.Dropout(dropout_rate, name = 'dropout2')(dense1)
    
    # 全连接层2
    outputs = keras.layers.Dense( num_classes, 
                                  activation='softmax',
                                  kernel_regularizer= keras.regularizers.l2(regularizers_lambda),
                                  bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                  name = 'outputs')(pool_outputs)
    
    model = keras.Model(inputs = inputs, outputs = outputs)

    return model

if __name__ == "__main__":
    # textCNN_Model = TextCNN(settings.max_sen_len, settings.embedding_dim, 
    #                         settings.filters_size, settings.num_filters, settings.drop_keep_prob, 
    #                         settings.n_class, settings.regularizers_lambda)
    # textCNN_Model.summary()

    word2id = build_word2id()
    word2vecs2 = load_word2vec(settings.corpus_word2vec_path)
    # print(word2vecs2.shape)

    textCNN_Update_Model = TextCNN(settings.update_w2v, settings.max_sen_len, word2vecs2, settings.embedding_dim, 
                            settings.filters_size, settings.num_filters, settings.drop_keep_prob, 
                            settings.n_class, settings.regularizers_lambda)

    textCNN_Update_Model.summary()




