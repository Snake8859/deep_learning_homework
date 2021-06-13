'''
Author: snake8859
Date: 2021-06-05 15:33:43
LastEditTime: 2021-06-06 18:46:05
LastEditors: Please set LastEditors
Description: rnn 网络模型
FilePath: \code\model.py
'''
import tensorflow as tf
import settings

# 构建模型
rnn_model = tf.keras.Sequential([
    # 不定长度的输入
    tf.keras.layers.Input((None,)), # [batch, seq_len]
    # 词嵌入层 '''Embedding的理解：https://www.jiqizhixin.com/articles/2019-03-27-7'''
    tf.keras.layers.Embedding(input_dim=settings.VOCAB_SIZE, output_dim=128), # [batch, seq_len] => [batch, seq_len, embed_dim]
    # 第一个LSTM层，返回序列作为下一层的输入 
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True), # [batch, seq_len, embed_dim] => [batch, seq_len, lstm1_dim]
    # 第二个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True), # [batch, seq_len, lstm1_dim] => [batch, seq_len, lstm2_dim]
    # 对每一个时间点的输出都做softmax，预测下一个词的概率 '''TimeDistributed的理解：https://blog.csdn.net/u012193416/article/details/79477220 '''
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(settings.VOCAB_SIZE, activation='softmax')),  # [batch, seq_len, vocab_size] | 理解：seq_len长度的每个词的在字典(vocab)概率分布 
])

# 查看模型结构
rnn_model.summary()
# 配置优化器和损失函数
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)
# rnn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.1), loss=tf.keras.losses.categorical_crossentropy)