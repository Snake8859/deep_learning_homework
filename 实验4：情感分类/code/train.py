'''
Author: snake8859
Date: 2021-06-20 14:47:55
LastEditTime: 2021-06-22 11:28:07
LastEditors: Please set LastEditors
Description: 模型训练
FilePath: \code\train.py
'''
import os
import datetime
from model import TextCNN
from data_process import *
import tensorflow as tf
# 管理GPU内存
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train(word2vecs2, emotion_train_data_loader, emotion_valid_data_loader):
    '''
    @description: 网络模型训练入口
    @param 
        word2vecs2 预训练词向量
        emotion_train_data_loader 情感训练数据加载器
        emotion_valid_data_loader 情感验证数据数据加载器
    @return 
        text_cnn 网络模型
    '''    
    text_cnn = TextCNN( settings.update_w2v, settings.max_sen_len, word2vecs2, settings.embedding_dim, 
                        settings.filters_size, settings.num_filters, settings.drop_keep_prob, 
                        settings.n_class, settings.regularizers_lambda)

    text_cnn.summary()

    # 测试
    # for x, y in emotion_train_data_loader.__iter__():
    #     print(x.shape)
    #     pred_y = text_cnn(x)
    #     print(pred_y.shape)
    #     print(y.shape)
    #     break

    # 编译模型
    text_cnn.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate= settings.learning_rate), # Adma优化器
        loss = tf.keras.losses.binary_crossentropy, # 二值交叉熵损失
        metrics= [tf.keras.metrics.binary_accuracy] # 二值评估器
    )
    
    dir_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 创建TensorBoard回调函数
    fit_log_dir = settings.log_dir + dir_datetime
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = fit_log_dir, # 输出路径
        histogram_freq = 1, # 统计每层直方图
        profile_batch = 0, # 不启动profile
        update_freq = 'epoch' # 更新频次，以batch
    )
    os.mkdir(fit_log_dir) # 创建目录

    # 创建Checkpoints回调函数
    checkpoint_dir = settings.check_dir + dir_datetime + '.ckpt'
    # os.mkdir(checkpoint_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    # 开始训练
    history = text_cnn.fit(
                emotion_train_data_loader.for_fit(), 
                steps_per_epoch=emotion_train_data_loader.steps, 
                epochs= settings.n_epoch,
                callbacks=[tensorboard_callback, cp_callback],
                validation_data = emotion_valid_data_loader.for_fit(),
                validation_steps = emotion_valid_data_loader.steps)

    print(history.history)

    # 保存模型
    print("\nSaving model...")
    out_save_dir =  settings.save_dir + dir_datetime
    os.mkdir(out_save_dir)
    # tf.keras.models.save_model(text_cnn, out_save_dir)
    tf.saved_model.save(text_cnn, out_save_dir)

    return text_cnn

def data_read(data_path):
    '''
    @description: 数据读取器
    @param 
        data_path 数据路径
    @return 
        emotion_data 情感数据
    '''    
    emotion_data = []
    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            sp = line.strip().split()
            # print(len(sp[1:]))
            if(len(sp[1:]) <= settings.max_sen_len): # 筛选长度小于75的句子
                emotion_data.append(sp)
    return emotion_data


if __name__ == "__main__":
    
    word2id = build_word2id() # 字典映射序号
    word2vecs = build_word2vec(settings.pre_word2vec_path, word2id, settings.corpus_word2vec_path) # 字典映射向量 
    # word2vecs = load_word2vec(settings.corpus_word2vec_path)
    # 情感分类训练数据
    train_emotion_data = data_read(settings.train_path)
    # 情感分类训练数据加载器
    # emotion_train_data_loader = EmotionDataLoader(train_emotion_data, settings.batch_size,word2id, word2vecs, random= True)
    emotion_train_data_loader = EmotionDataLoader1(train_emotion_data, settings.batch_size,word2id, random= True)
    
    # 情感分类验证数据
    valid_emotion_data = data_read(settings.dev_path)
    
    # 情感分类验证数据加载器
    # emotion_valid_data_loader = EmotionDataLoader(valid_emotion_data, settings.batch_size, word2id, word2vecs, random= False)
    emotion_valid_data_loader = EmotionDataLoader1(valid_emotion_data, settings.batch_size, word2id, random= False)

    text_cnn = train(word2vecs, emotion_train_data_loader, emotion_valid_data_loader)
    