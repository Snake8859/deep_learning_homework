'''
Author: snake8859
Date: 2021-06-06 15:06:33
LastEditTime: 2021-06-13 14:21:32
LastEditors: Please set LastEditors
Description: 写诗工具类
FilePath: /code/utils.py
'''
import numpy as np
import settings

def generate_random_poetry(word2ix, ix2word, model, s=settings.START_FLAG):
    '''
    随机生成一首诗
    :param word2ix: 文字映射序号
    :param ix2word: 序号映射文本
    :param model: 用于生成古诗的模型
    :param s: 用于生成古诗的起始字符串，默认为空串
    :return: 一个字符串，表示一首古诗
    '''
    # 将初始字符串转成token
    if s != settings.START_FLAG:
        token_ids = [settings.START_FLAG, s]
    else:
        token_ids = [s]
    while len(token_ids) < settings.MAX_LEN:
        # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[START] </s>的概率分布
        # _probas = model.predict([token_ids, ]) # (1, seq_len, 8293)
        _probas = model.predict([token_ids, ])[0, -1, :-1] # (8292, )
        # print(_probas)
        # 按照出现概率，对所有token倒序排列，取前100
        p_args = _probas.argsort()[::-1][:100]
        # 排列后的概率顺序
        p = _probas[p_args]
        # 先对概率归一
        p = p / sum(p)
        # 再按照预测出的概率，随机选择一个词作为预测结果
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index]
        # 保存
        token_ids.append(target)
        if target == settings.EOP_FLAG:
            break
        # print(token_ids)

    # 解码诗句
    poetry = ''
    for i in token_ids:
        poetry_word = ix2word.item()[i] # loaded_dict，使用.item（）方法访问字典。
        poetry += poetry_word
            
    return poetry


def generate_acrostic(word2ix, ix2word, model, head):
    '''
    随机生成一首藏头诗
    :param word2ix: 文字映射序号
    :param ix2word: 序号映射文本
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    '''
    # 使用空串初始化token_ids，加入[START]
    token_ids = [settings.START_FLAG]
    # 标点符号，这里简单的只把逗号和句号作为标点
    punctuation_ids = [settings.COMMA_FLAT, settings.FULL_STOP_FLAT]
    # 缓存生成的诗的list
    poetry = []
    # 对于藏头诗中的每一个字，都生成一个短句
    for ch in head:
        # 先记录下这个字
        poetry.append(ch)
        # 将藏头诗的字符转成token id
        token_id = word2ix.item()[ch]
        # 加入到列表中去
        token_ids.append(token_id)
        # 开始生成一个短句
        while True:
            # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[START] </s>的概率分布
            _probas = model.predict([token_ids, ])[0, -1, :-1]
            # 按照出现概率，对所有token倒序排列，取前100
            p_args = _probas.argsort()[::-1][:100]
            # 排列后的概率顺序
            p = _probas[p_args]
            # 先对概率归一
            p = p / sum(p)
            # 再按照预测出的概率，随机选择一个词作为预测结果
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index]
            # 保存
            token_ids.append(target)
            # 只有不是特殊字符时，才保存到poetry里面去
            if target < 8290:
                poetry.append(ix2word.item()[target])
                # print(poetry)
            if target in punctuation_ids:
                break
    return ''.join(poetry)