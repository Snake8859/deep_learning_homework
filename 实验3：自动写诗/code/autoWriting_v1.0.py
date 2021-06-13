'''
Author: snake8859
Date: 2021-06-04 15:55:03
LastEditTime: 2021-06-13 15:02:24
LastEditors: Please set LastEditors
Description: RNN 自动写诗测试v1.0
FilePath: \code\autoWriting_v1.0.py
'''
import os
import time
import random
import datetime
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 警告抑制
from tensorflow.python.keras.backend import rnn
import settings
import utils
from dataset_loader import PoetryDataGenerator
from model import rnn_model

def data_view(data_path):
    tang_ds = np.load(data_path, allow_pickle=True)
    # print(tang_ds.files) # ['ix2word', 'word2ix', 'data']

    ix2word = tang_ds['ix2word'] # 序号对应文本
    word2ix = tang_ds['word2ix'] # 文本对应序号
    data = tang_ds['data'] # 57580首唐诗，每首诗限定在125词，不足125词的以</s>填充

    print('ix2word : {0}'.format(len(ix2word.item()))) # 8293 字典大小
    print(ix2word)
    print('word2ix : {0}'.format(len(word2ix.item()))) # 8293 字典大小
    # print(word2ix)
    # print(word2ix.item()['，'], word2ix.item()['。'])
    print('data : {0}'.format(data.shape)) # (57580, 125) 

    for poetry in data:
        '''
            <START> 8291
            <EOP> 8290
        '''
        # print(poetry.shape)
        # print(poetry)
        for i in poetry:
            poetry_word = ix2word.item()[i] # loaded_dict，使用.item（）方法访问字典。
            if poetry_word != '</s>':
                # print(poetry_word, i, end=' ')
                print(poetry_word, end='')
        print('\n')
        time.sleep(0.3)
        # break


def prepareData(data):

    poetry_list = []
    for poetry in data:
        # print(poetry)
        poetry_ids = []
        for i in poetry:
            if i != 8292: # 去掉填充<s/>
                poetry_ids.append(i)
        poetry_list.append(poetry_ids)
    
    poetry_data = np.array(poetry_list)
    # print(poetry_data.shape) # (57588,)
    p_ds = PoetryDataGenerator(poetry_data)
    # 测试数据输出
    # for x_data, y_data in p_ds:
        # print(x_data.shape, y_data.shape)
        # for i in x_data[0]:
        #     poetry_x = ix2word.item()[i] # loaded_dict，使用.item（）方法访问字典。
        #     print(poetry_x, end='')
            
        # print('\n')
        
        # for j in y_data[0]:
        #     poetry_y = ix2word.item()[j] # loaded_dict，使用.item（）方法访问字典。
        #     print(poetry_y, end='')
        # break
    
    return p_ds
    
    '''
    poetrys_x = [] # 数据(x)
    poetrys_y = [] # 标签(y)

    for poetry in data:
        # print(poetry)
        poetry_content = ''
        poetry_ids = []
        for i in poetry:
            if i != 8292: # 去掉填充<s/>
                poetry_ids.append(i)
            
        x_data = poetry_ids[:-1]
        y_data = poetry_ids[1:]

        # print(x_data, len(x_data))
        # print(y_data, len(y_data))

        # for i in x_data:
        #     poetry_x = ix2word.item()[i] # loaded_dict，使用.item（）方法访问字典。
        #     print(poetry_x, end='')
        # print('\n')
        # for j in y_data:
        #     poetry_y = ix2word.item()[j] # loaded_dict，使用.item（）方法访问字典。
        #     print(poetry_y, end='')

        poetrys_x.append(x_data) # 数据(x)：提取古诗主体内容    
        poetrys_y.append(tf.one_hot(y_data, settings.VOCAB_SIZE)) # 标签(y)：错开一位的x，并添加one-hot编码s
        
    poetry_ds = tf.data.Dataset.from_tensor_slices((poetrys_x, poetrys_y)).batch(settings.BATCH_SIZE).shuffle(settings.BUFFER_SIZE)

    for x, y in poetry_ds:
        print(x.shape, y.shape)
        poetry_x = x.numpy()[0]
        poetry_y = y.numpy()[0]
        for i in poetry_x:
            poetry_x = ix2word.item()[i] # loaded_dict，使用.item（）方法访问字典。
            print(poetry_x, end='')
        print('\n')
        for j in poetry_y:
            poetry_y = ix2word.item()[j] # loaded_dict，使用.item（）方法访问字典。
            print(poetry_y, end='')
        break
    
    return poetry_ds
    '''


class Evaluate(tf.keras.callbacks.Callback):
    """
    在每个epoch训练完成后，保留最优权重，并随机生成settings.SHOW_NUM首古诗展示
    """

    def __init__(self):
        super().__init__()
        # 给loss赋一个较大的初始值
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch训练完成后调用
        # 如果当前loss更低，就保存当前模型参数
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            rnn_model.save(settings.BEST_MODEL_PATH)
        # 随机生成几首古体诗测试，查看训练效果
        print()
        for i in range(settings.SHOW_NUM):
            # 随机生成一个序号
            random_id = random.randint(0, settings.VOCAB_SIZE)
            print(utils.generate_random_poetry(word2ix, ix2word, rnn_model, random_id))


if __name__ == "__main__":
    data_path = './tang.npz'
    # data_view(data_path)
    # exit()

    tang_ds = np.load(data_path, allow_pickle=True)
    ix2word = tang_ds['ix2word'] # 序号对应文本
    word2ix = tang_ds['word2ix'] # 文本对应序号
    data = tang_ds['data'] # 57580首唐诗，每首诗限定在125词，不足125词的以</s>填充

    poetry_ds = prepareData(data)

    # # 创建TensorBoard回调函数
    # dir_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # fit_log_dir = './logs/' + dir_datetime
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir = fit_log_dir, # 输出路径
    #     histogram_freq = 1, # 统计每层直方图
    #     profile_batch = 0, # 不启动profile
    #     update_freq = 'epoch' # 更新频次，以batch
    # )
    # os.makedirs(fit_log_dir) # 创建目录

    # 开始训练
    # rnn_model.fit_generator(poetry_ds.for_fit(), steps_per_epoch=poetry_ds.steps, epochs=settings.TRAIN_EPOCHS, callbacks=[Evaluate(), tensorboard_callback])


    # 测试
    rnn_model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)
    
    # 指定和随机诗句
    # for i in range(settings.SHOW_NUM):
    #     word_id = word2ix.item()['春'] # 指定某个字开头
    #     print(utils.generate_random_poetry(word2ix, ix2word, rnn_model, word_id))
        # random_id = random.randint(0, settings.VOCAB_SIZE) # 随机生成
        # print(utils.generate_random_poetry(word2ix, ix2word, rnn_model))
    
    # 藏头诗句
    # head = ['金','榜','题','名']
    head = list('深度学习自动写诗')
    # # print(head)
    for i in range(settings.SHOW_NUM):
        print(utils.generate_acrostic(word2ix, ix2word, rnn_model, head))


