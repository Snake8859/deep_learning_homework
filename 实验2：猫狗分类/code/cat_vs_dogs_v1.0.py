'''
Author: snkae8859
Date: 2021-04-21 11:40:04
LastEditTime: 2021-06-02 15:43:25
LastEditors: Please set LastEditors
Description: 
    猫狗分类v2.1：
        1. 网络架构： 自定义[CNN + MLP] (Sequential API方式)
        2. 采用model方式训练和测试
        3. 自定义数据加载器
    问题：
        1.模型在训练集上表现好(90%以上)，在测试集上一般(70%左右)，出现过拟合
        2.添加Dropout策略防止过拟合，但是模型性能瓶颈(80%左右)
FilePath: \code\cat_vs_dogs_v1.0.py
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
        基础模型
        ? 过拟合 Dropout
        result:
           - epoch20: 训练集-0.99 测试集-0.7689 过拟合
    '''
    # model = create_model1()
    
    
    '''
        改良模型
         引入Dropout或Batch Normalization
         ? 会陷入局部解
        reuslt:
         - epoch20 -> 训练集-0.830  测试集-0.8126
         - cpoch30 -> 训练集-0.926  测试集-0.8283
    '''
    model = create_model2()

    # exit()


    # 编译模型
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate, momentum=0.001), #  优化器 （Adma:局部最优） （SGD） 
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

    # 执行模型
    model.fit(
        train_image_label_ds, # 训练数据
        validation_data= valid_image_label_ds, # 验证数据
        epochs = num_epochs, # 训练轮数
        callbacks = [tensorboard_callback] # tensorboard回调函数
    )
    

    # 保存模型
    fit_save_dir = save_dir + dir_datetime
    os.mkdir(fit_save_dir)
    tf.saved_model.save(model, fit_save_dir)

    return model


def test(test_file_list, buffer_size, batch_size, num_epoch, my_model = None):
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
        # base_epoch20 = os.path.join(save_dir, '20210602-101104') # 基础版 epoch20 过拟合
        imporve_epoch20 = os.path.join(save_dir, '20210602-110422') # 改良版 epoch20 正常
        # imporve_epoch30 = os.path.join(save_dir, '20210602-112038') # 改良版 epoch30 正常（性能瓶颈）
        my_model = tf.saved_model.load(imporve_epoch20)


    # 实例化测试集数据加载器
    test_dataLoader = CatDogLoader(test_file_list, buffer_size, batch_size, num_epoch)

    # 实例化评估器
    sparse_categorical_accuracy =  tf.keras.metrics.SparseCategoricalAccuracy()
    test_image_lable_ds = test_dataLoader.image_label_ds
    
    # 测试
    for images, lables in test_image_lable_ds:
        # print(images.shape, lables.shape)
        # 模型预测
        lables_pred = my_model(images)
        sparse_categorical_accuracy.update_state(
             y_true = lables,
             y_pred = lables_pred
        )
        # print(lable)
        # print(lable_pred)
        # break
    print('test accuracy: {0}'.format(sparse_categorical_accuracy.result()))
    return sparse_categorical_accuracy.result()


def create_model1():

    model = tf.keras.Sequential([
        # 第1卷积和池化层
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=2, activation=tf.nn.relu, input_shape = (224, 224, 3)),
        tf.keras.layers.MaxPool2D(),

        # 第2卷积和池化层
        tf.keras.layers.Conv2D(filters=64,kernel_size=5, strides=2, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 第3卷积和池化层
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 第4卷积和池化层
        tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 特征拉直
        tf.keras.layers.Flatten(), 

        # 5,6全连接层
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    print(model.summary())

    return model


def create_model2():

    model = tf.keras.Sequential([
        # 第1卷积和池化层
        tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=2, activation=tf.nn.relu, input_shape = (224, 224, 3)),
        tf.keras.layers.MaxPool2D(),

        # 第2卷积和池化层
        tf.keras.layers.Conv2D(filters=64,kernel_size=5, strides=2, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 第3卷积和池化层
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 第4卷积和池化层
        tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),

        # 特征拉直
        tf.keras.layers.Flatten(), 

        # 5,6全连接层
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    print(model.summary())

    return model


    
if __name__ == '__main__':
    num_epochs = 30 # 训练轮数
    batch_size = 10 # 10图片为一组
    learning_rate = 0.01
    save_dir = './save/v1.0/'
    log_dir = './tensorboard/v1.0/'

    # 训练数据集
    train_root = './dogs-vs-cats/train/'
    train_file_list = [ os.path.join(train_root, file_name) for file_name in os.listdir(train_root) ]
   

    # 验证数据集
    valid_root = './dogs-vs-cats/valid'
    valid_file_list = [ os.path.join(valid_root, file_name) for file_name in os.listdir(valid_root) ]

    # 网络训练
    my_model = train(train_file_list, valid_file_list, 
                  buffer_size= 10000, batch_size = batch_size, learning_rate = learning_rate, 
                  num_epochs = num_epochs, save_dir= save_dir, log_dir= log_dir)

    # 测试数据集
    test_root = './dogs-vs-cats/test/'
    test_file_list = [ os.path.join(test_root, file_name) for file_name in os.listdir(test_root) ]

    test(test_file_list, buffer_size = 3000, batch_size = 1, num_epoch = 1, my_model = my_model)
    # test(test_file_list, buffer_size = 3000, batch_size = 1, num_epoch = 1)



    