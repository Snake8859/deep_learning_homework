'''
Author: your name
Date: 2021-06-05 14:46:40
LastEditTime: 2021-06-12 16:14:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \code\AaronJny_Code\settings.py
'''


# 训练的batch size
BATCH_SIZE = 16
# 数据的buffer_size
BUFFER_SIZE = 50000
# 句子最大长度
MAX_LEN = 64
# 数据集路径
DATASET_PATH = './tang.npz'
# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 5
# 共训练多少个epoch
TRAIN_EPOCHS = 20
# 最佳权重保存路径
BEST_MODEL_PATH = './best_model.h5'
# 字典长度
VOCAB_SIZE = 8293
# 开始标识
START_FLAG = 8291
# 结束标识
EOP_FLAG = 8290
# 逗号标识
COMMA_FLAT = 7066
# 句号标识
FULL_STOP_FLAT = 7435