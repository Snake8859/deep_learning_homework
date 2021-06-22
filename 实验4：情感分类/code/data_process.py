'''
Author: your name
Date: 2021-06-19 16:38:30
LastEditTime: 2021-06-22 10:50:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \code\data_process.py
'''
import settings
import gensim
import numpy as np
import math

def build_word2id(file = None):
    """
    :param file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./data/train.txt', './data/validation.txt', './data/test.txt']
    # print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    # print(len(word2id)) # 58954

    if file :
        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w+'\t')
                f.write(str(word2id[w]))
                f.write('\n')
    
    return word2id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        np.save(save_to_path, word_vecs)
    return word_vecs


def load_word2vec(save_path):
    '''
    @description: 加载词向量
    @param {*}
    @return {*}
    '''    
    return np.load(save_path)


class EmotionDataLoader:
    '''情感数据加载器（废弃）'''
    def __init__(self, data, batch_size, word2id, word2vecs, random=False):
        # 数据集
        self.data = np.array(data, dtype='object')
        # 字符对映序号
        self.word2id = word2id
        # 预训练Embedding
        self.word2vecs = word2vecs
        # batch size
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size)) #
        # 每个epoch开始时是否随机混洗
        self.random = random
    
    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data) # 数据长度
        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for i in range(0, self.steps):
            batch_data = self.data[i * self.batch_size : i * self.batch_size + self.batch_size, ]
            # print(batch_data.shape, i * self.batch_size + self.batch_size)
            # 填充到相同的长度
            batch_data = self.sequence_padding(batch_data)
            # print(batch_data.shape)
            # print(batch_data[0])
            # 制作训练数据和标签
            x = np.zeros(shape = (self.batch_size, settings.max_sen_len, settings.embedding_dim), dtype= np.float32)
            y = np.zeros(shape = (self.batch_size, 2), dtype = np.uint8)
            for i, data in enumerate(batch_data):
                vecs = []
                for word in data[1:]:
                    vecs.append(self.word2vecs[self.word2id[word]])
                vecs = np.array(vecs)
                # print(vecs.shape)
                x[i] = vecs
                y[i] = np.eye(settings.n_class, dtype = np.uint8)[int(data[0])]
            # print(x, x.shape)
            # print(y, y.shape)
            # break
            yield x, y
            del batch_data

    
    def sequence_padding(self, data, length = settings.max_sen_len, padding ='_PAD_'):
        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line[1:])
            # print(padding_length)
            if padding_length >= 0: # 不足填充
                outputs.append(np.concatenate([line, [padding] * padding_length]))
        return np.array(outputs)


    def for_fit(self):
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()


class EmotionDataLoader1:
    '''情感数据加载器'''
    def __init__(self, data, batch_size, word2id, random=False):
        # 数据集
        self.data = np.array(data, dtype='object')
        # 字符对映序号
        self.word2id = word2id
        # batch size
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size)) #
        # 每个epoch开始时是否随机混洗
        self.random = random
    
    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data) # 数据长度
        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for i in range(0, self.steps):
            batch_data = self.data[i * self.batch_size : i * self.batch_size + self.batch_size, ]
            # print(batch_data.shape, i * self.batch_size + self.batch_size)
            # 填充到相同的长度
            batch_data = self.sequence_padding(batch_data)
            # print(batch_data.shape)
            # print(batch_data[0])
            # 制作训练数据和标签
            x = np.zeros(shape = (self.batch_size, settings.max_sen_len), dtype= np.uint16)
            y = np.zeros(shape = (self.batch_size, 2), dtype = np.uint8)
            for i, data in enumerate(batch_data):
                ids = []
                for word in data[1:]:
                    ids.append(self.word2id[word])
                ids = np.array(ids)
                # print(vecs.shape)
                x[i] = ids
                y[i] = np.eye(settings.n_class, dtype = np.uint8)[int(data[0])]
            # print(x, x.shape)
            # print(y, y.shape)
            # break
            yield x, y
            del batch_data

    
    def sequence_padding(self, data, length = settings.max_sen_len, padding ='_PAD_'):
        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line[1:])
            # print(padding_length)
            if padding_length >= 0: # 不足填充
                outputs.append(np.concatenate([line, [padding] * padding_length]))
        return np.array(outputs)


    def for_fit(self):
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()


def data_view(data_path):
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            print(''.join(line[1:]).strip(), line[0])
            print('\n')


if __name__ == "__main__":

    # data_view(settings.train_path)
    # exit()

    word2id = build_word2id()
    # word2vecs1 = build_word2vec(settings.pre_word2vec_path, word2id, settings.corpus_word2vec_path)
    # print(word2vecs1.shape) # (59290, 50)
    word2vecs2 = load_word2vec(settings.corpus_word2vec_path)
    # print(word2vecs2.shape)

    emotion_data = []
    with open(settings.train_path, encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            sp = line.strip().split()
            # print(len(sp[1:]))
            if(len(sp[1:]) <= settings.max_sen_len): # 筛选长度小于75的句子
                emotion_data.append(sp)
    
    # # print(len(emotion_data)) # 19897
    # # print(emotion_data[0])
    emotion_data_loader = EmotionDataLoader1(emotion_data, 1, word2id, word2vecs2, random= True)
    # print(emotion_data_loader.__iter__()) # 创建生成器

    for x, y in emotion_data_loader.__iter__():
        print(x, x.shape)
        print(y, y.shape)
        break