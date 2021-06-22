'''
Author: snake8859
Date: 2021-06-19 16:44:54
LastEditTime: 2021-06-22 11:31:13
LastEditors: Please set LastEditors
Description: 配置信息
FilePath: \code\settings.py
'''
update_w2v = True           # 是否在训练中更新w2v
vocab_size = 59290          # 词汇量，与word2id中的词汇量一致
n_class = 2                 # 分类数：分别为pos和neg
max_sen_len = 75            # 句子最大长度
embedding_dim = 50          # 词向量维度（卷积核宽度）
filters_size = '3,4,5'      # 卷积核宽度；text-cnn 选择3，4，5
batch_size = 50             # 批处理尺寸
n_epoch = 5                # 训练迭代周期，即遍历整个训练样本的次数
learning_rate = 0.001       # 学习率；若opt=‘adadelta'，则不需要定义学习率
drop_keep_prob = 0.8        # dropout层，参数keep的比例
regularizers_lambda = 0.01  # 正则化比例
num_filters = 256           # 卷积层filter的数量
save_dir = './saves/'       # 训练模型保存的地址
check_dir = './checkspoint/' # 训练权重保存的地址
log_dir = './logs/'         # 训练日志保存的地址

train_path = './data/train.txt'
dev_path = './data/validation.txt'
test_path = './data/test.txt'
word2id_path = './data/word_to_id.txt'
pre_word2vec_path = './data/wiki_word2vec_50.bin'
# corpus_word2vec_path = './data/corpus_word2vec.txt'
corpus_word2vec_path = './data/corpus_word2vec.npy'
