'''
Author: snake8859
Date: 2021-06-20 15:41:13
LastEditTime: 2021-06-22 16:43:24
LastEditors: Please set LastEditors
Description: 测试
FilePath: \code\test.py
'''
from data_process import *
from model import TextCNN
import tensorflow as tf
# 管理GPU内存
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def test(text_cnn, emotion_test_data_loader):
    # text_cnn.summary()

    # 实例化评估器
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()

    for x, y in emotion_test_data_loader.__iter__():
        # print(x.shape)
        # x = x.astype(np.float32)
        y_pred = text_cnn(x, training = False)
        # print(x, x.shape)
        # print(y, y_pred)
        binary_accuracy.update_state(
            y_true = y,
            y_pred = y_pred
        )
        # break

    print('test accuracy: {0}'.format(binary_accuracy.result()))


if __name__ == "__main__":
    word2id = build_word2id() # 字典映射序号
    # 情感分类测试数据
    test_emotion_data = []
    with open(settings.test_path, encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            sp = line.strip().split()
            # print(len(sp[1:]))
            if(len(sp[1:]) <= settings.max_sen_len): # 筛选长度小于75的句子
                test_emotion_data.append(sp)
    # 情感分类测试数据加载器
    emotion_test_data_loader = EmotionDataLoader1(test_emotion_data, 1, word2id, random = False)
    
    save_dir = settings.save_dir + '20210622-105935'

    # 加载模型(Saved_Model)
    # text_cnn = tf.keras.models.load_model(save_dir)
    text_cnn = tf.saved_model.load(save_dir)

    # 加载模型(Checkpoints)
    # text_cnn = TextCNN(settings.max_sen_len, settings.embedding_dim, 
    #         settings.filters_size, settings.num_filters, settings.drop_keep_prob, 
    #         settings.n_class, settings.regularizers_lambda)
    
    # text_cnn.load_weights(settings.check_dir + '20210620-195928.ckpt')
    test(text_cnn, emotion_test_data_loader)