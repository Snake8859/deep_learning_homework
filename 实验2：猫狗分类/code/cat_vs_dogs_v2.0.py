'''
Author: snake8859
Date: 2021-04-21 11:40:04
LastEditTime: 2021-06-02 16:49:04
LastEditors: Please set LastEditors
Description: 
    猫狗分类v2.0：
        1. 网络架构： 预训练InceptionV3 + MLP (Functional API方式)
        2. 采用配置式训练和测试
        3. 自定义数据加载器
    问题：暂无问题，正确率80%以上，待继续提升
FilePath: \code\cat_vs_dogs_v2.0.py
'''

import os
import datetime
import tensorflow as tf
from data_loader import CatDogLoader
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # CPU运行


def train(train_file_list, valid_file_list, buffer_size, batch_size, learning_rate, num_epochs, save_dir, log_dir):
    '''
    @description: 模型训练
    @param 
        train_file_list 训练数据集文件
        valid_file_list 验证数据集文件
        buffer_size 缓冲区大小
        batch_size 批次大小
        num_epoch 重复迭代次数
        save_dir 模型路径
        log_dir 日志路径
    @return 
        model 训练模型
    '''   

    # 实例化训练集数据加载器
    train_dataLoader = CatDogLoader(train_file_list, buffer_size, batch_size, 1)
    train_image_label_ds = train_dataLoader.image_label_ds

    # 实例化验证集数据加载器
    valid_dataLoader = CatDogLoader(valid_file_list, buffer_size = 2000, batch_size = 1, num_epoch= 1)
    valid_image_label_ds = valid_dataLoader.image_label_ds

    '''
        VGG-16
           SGD(0.01) epcho 3 训练集-0.923 测试集-0.9776
    '''
    # my_cats_vs_dogs_model = vgg_16_Model()


    '''
        InceptionV3
            SGD(0.01) epcho 3 训练集-0.989 测试集-0.982
    '''
    # my_cats_vs_dogs_model = inception_v3_model()

    
    '''
        ResNet-50
            SGD(0.01) epcho 3 训练集-0.986 测试集-0.982
    '''
    my_cats_vs_dogs_model = res_50_model()

    # exit()

    # 编译模型
    my_cats_vs_dogs_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.001), # 优化器 （Adma：未学习）
        loss = tf.keras.losses.sparse_categorical_crossentropy, # 损失函数 （交叉熵损失）
        metrics = [tf.keras.metrics.sparse_categorical_accuracy] # 评估器
    )

    # 创建TensorBoard回调函数
    dir_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fit_log_dir = log_dir + dir_datetime
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = fit_log_dir, # 输出路径
        histogram_freq = 1, # 统计每层直方图
        profile_batch = 0, # 不启动profile
        update_freq = 'epoch' # 更新频次，以batch
    )
    os.makedirs(fit_log_dir) # 创建目录

    # 训练模型
    my_cats_vs_dogs_model.fit(
        train_image_label_ds, # 训练数据
        validation_data= valid_image_label_ds, # 验证数据
        epochs = num_epochs, # 训练轮数
        callbacks = [tensorboard_callback] # tensorboard回调函数
    )

    # 保存模型
    out_save_dir = save_dir + dir_datetime
    os.mkdir(out_save_dir)
    tf.saved_model.save(my_cats_vs_dogs_model, out_save_dir)

    return my_cats_vs_dogs_model


def test(test_file_list, buffer_size, batch_size, my_model = None):
    '''
    @description: 网络模型测试
    @param 
        test_file_list 测试数据集文件
        buffer_size 缓冲区大小
        batch_size 批次大小
        num_epoch 重复迭代次数
        my_model 模型
    @return 
        test_result 测试结果
    '''

    if my_model == None: # 保存路径中加载模型
        # vgg16_epoch3 = os.path.join(save_dir, '20210602-155339') # VGG-16 优秀
        inception_v3_epoch3 = os.path.join(save_dir, '20210602-161343') # inceptionV3 优秀
        my_model = tf.saved_model.load(inception_v3_epoch3)

    # 加载数据
    test_dataLoader = CatDogLoader(test_file_list, buffer_size, batch_size, 1)
    test_image_lable_ds = test_dataLoader.image_label_ds

    # 实例化评估器
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # 测试
    for images, lables in test_image_lable_ds:
        # 模型预测
        lables_pred = my_model(images)
        sparse_categorical_accuracy.update_state(
             y_true = lables,
             y_pred = lables_pred
        )

    print('test accuracy: {0}'.format(sparse_categorical_accuracy.result()))

    return sparse_categorical_accuracy.result()


def vgg_16_Model():
    inputs_tensor = tf.keras.Input(shape = (224, 224, 3))
    vgg_net_model = tf.keras.applications.VGG16(
        include_top = False, # 是否包含全连接层
        weights = 'imagenet', # imagenet预训练权重
        pooling = 'avg', # 输出向量取平均，拉直
        input_tensor = inputs_tensor # 输入张量
    )
    # print(vgg_net_model.summary()) 

    # 使用Functional API方式建立层
    image_feature = vgg_net_model.output # VGGNet提取特征 [batch_size ,512]

    d1 = tf.keras.layers.Dense(units = 256, activation = tf.nn.relu)(image_feature) # 全连接第1层 [batch_size, 256]
    d2 = tf.keras.layers.Dense(units = 64, activation = tf.nn.relu)(d1) # 全连接第2层 [batch_size, 64]
    outputs = tf.keras.layers.Dense(units = 2, activation = tf.nn.softmax)(d2) # 全连接第3层 [batch_size, 2]

    # 建立模型
    my_cats_vs_dogs_model = tf.keras.Model(
        inputs = vgg_net_model.inputs,
        outputs = outputs
    )

    # 打印网络形状
    print(my_cats_vs_dogs_model.summary())

    return my_cats_vs_dogs_model


def inception_v3_model():
    inputs_tensor = tf.keras.Input(shape = (224, 224, 3))
    # 使用预训练的GoogleNet模型
    goolge_net_model = tf.keras.applications.InceptionV3(
        include_top = False, # 是否包含全连接层
        weights = 'imagenet', # imagenet预训练权重
        pooling = 'avg', # 输出向量取平均，拉直
        input_tensor = inputs_tensor # 输入张量
        # input_shape = (299, 299, 3) # 输入张量的形状
    )

    # print(goolge_net_model.summary())

    # 使用Functional API方式建立层
    image_feature = goolge_net_model.output # GoogleNet提取特征 [batch_size ,2048]
    
    d1 = tf.keras.layers.Dense(units = 512, activation = tf.nn.relu)(image_feature) # 全连接第1层 [batch_szie, 512]
    d2 = tf.keras.layers.Dense(units = 128, activation = tf.nn.relu)(d1) # 全连接第2层 [batch_szie, 128]
    outputs = tf.keras.layers.Dense(units = 2, activation = tf.nn.softmax)(d2) # 全连接第3层 [batch_szie, 2]

    # 建立模型
    my_cats_vs_dogs_model = tf.keras.Model(
        inputs = goolge_net_model.inputs,
        outputs = outputs
    )

    # 打印网络形状
    print(my_cats_vs_dogs_model.summary())

    return my_cats_vs_dogs_model


def res_50_model():
    inputs_tensor = tf.keras.Input(shape = (224, 224, 3))
    res_net_model = tf.keras.applications.ResNet50(
        include_top = False, # 是否包含全连接层
        weights = 'imagenet', # imagenet预训练权重
        pooling = 'avg', # 输出向量取平均，拉直
        input_tensor = inputs_tensor # 输入张量
    )

    # print(res_net_model.summary())

    # 使用Functional API方式建立层
    image_feature = res_net_model.output # GoogleNet提取特征 [batch_size ,2048]
    
    d1 = tf.keras.layers.Dense(units = 512, activation = tf.nn.relu)(image_feature) # 全连接第1层 [batch_szie, 512]
    d2 = tf.keras.layers.Dense(units = 128, activation = tf.nn.relu)(d1) # 全连接第2层 [batch_szie, 128]
    outputs = tf.keras.layers.Dense(units = 2, activation = tf.nn.softmax)(d2) # 全连接第3层 [batch_szie, 2]

    # 建立模型
    my_cats_vs_dogs_model = tf.keras.Model(
        inputs = res_net_model.inputs,
        outputs = outputs
    )

    # 打印网络形状
    print(my_cats_vs_dogs_model.summary())

    return my_cats_vs_dogs_model



if __name__ == '__main__':

    num_epochs = 3 # 训练轮数
    batch_size = 10 # 10图片为一组
    learning_rate = 0.01
    save_dir = './save/v2.0/'
    log_dir = './tensorboard/v2.0/'


    # 构建训练数据集2.0
    train_root = './dogs-vs-cats/train/'
    train_file_list = [os.path.join(train_root, file_name) for file_name in os.listdir(train_root) ]

    # 构建验证数据集2.0
    valid_root = './dogs-vs-cats/valid/'
    valid_file_list = [ os.path.join(valid_root, file_name) for file_name in os.listdir(valid_root) ]

    # 网络训练
    my_model = train(train_file_list, valid_file_list, 10000, batch_size, learning_rate, num_epochs, save_dir, log_dir)

    # 构建测试数据集2.0
    test_root = './dogs-vs-cats/test/'
    test_file_list = [ os.path.join(test_root, file_name) for file_name in os.listdir(test_root) ]

    # 网络测试
    test(test_file_list, buffer_size=3000, batch_size = 1, my_model=my_model)
    # test(test_file_list, buffer_size=3000, batch_size = 1)
    
    