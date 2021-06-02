import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import MNISTLoader
# 管理GPU内存
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 启动CPU运行


class CNN(tf.keras.Model):
    '''
        模型构建:
            卷积神经网络的结构和多层感知器结构类似，只是在多层感知器之前新加入一些卷积层和池化层。
            卷积层和池化层主要用于图像的特征提取，它是基于大脑的视觉皮层启发，引入感受野（Receptive Field）这一概念。
            【视觉皮层中的神经元并非与前一层的所有神经元相连，而只是感受这一片区域内的视觉信号，并只对局部区域的视觉刺激进行反应】
            CNN的卷积层正体现这一特性。
    '''
    def __init__(self):
        super().__init__()
        # 定义网络第1层：卷积层
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 32, # 卷积核个数
            kernel_size = [5, 5], # 卷积大小（感受野大小）
            padding = 'same', # padding策略(vaild 或 same)
            activation=tf.nn.relu # 激活函数
        )

        # 定义网络第2层：池化层
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size = [2, 2], # 池化核大小
            strides= 2 # 卷积步长
        )

        # 定义网络第3层：卷积层
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = [5,5],
            padding = "same",
            activation= tf.nn.relu
        )

        # 定义网络第4层：池化层
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size= [2,2],
            strides= 2
        )

        # 定义网络第5层：特征拉值
        self.flatten = tf.keras.layers.Reshape(target_shape = (7 * 7 * 64, ))


        # 定义网络第6层：全连接层
        self.dense1 = tf.keras.layers.Dense(
            units = 1024, # 神经元个数
            activation=tf.nn.relu # 激活函数
        )

        # 定义网络第7层：全连接层
        self.dense2 = tf.keras.layers.Dense(
            units = 64, 
            activation = tf.nn.relu
        )

        # 定义网络第8层：全连接层
        self.dense3 = tf.keras.layers.Dense(
            units = 10
        )
        

    def call(self, inputs):
        '''
        @description: 前向传播
        @param 
            inputs 输入向量
        @return 
            output 输出分类结果
        '''
        c1 = self.conv1(inputs) # [batch_size, 28, 28, 32]
        p1 = self.pool1(c1) # [batch_size , 14 ,14 ,32]
        c2 = self.conv2(p1) # [batch_size, 14, 14, 64]
        p2 = self.pool1(c2) # [batch_size, 7, 7, 64]
        f = self.flatten(p2) # [batch_size, 3136]
        d1 = self.dense1(f) # [batch_size, 1024]
        d2 = self.dense2(d1) # [batch_size, 64]
        d3 = self.dense3(d2) # [batch_size, 10]
        output = tf.nn.softmax(d3) # [batch_size, 10]
        return output


def train(num_epochs, batch_size, num_batches, learning_rates, log_dir, save_dir):
    '''
    @description: 模型训练
    @param 
        num_epochs 
        batch_size 
        num_batches 
        learning_rates
        log_dir 
        save_dir
    @return model
    '''
    # 模型实例化
    model = CNN()
    # 实例化模型优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Adam优化器
    # 实例化CheckPoint，设置保存模型对象
    check_point = tf.train.Checkpoint(myMode = model)
    # 实例化记录器
    summary_writer = tf.summary.create_file_writer(log_dir)

    # 迭代训练
    for batch_index in range(num_batches):
        x, y_true = dataLoader.get_batch(batch_size) # 每次迭代随机取batch_size个数据
        # print(x, x.shape, type(x))
        # print(y_true, y_true.shape, type(y_true))
        # break
        # 计算损失
        with tf.GradientTape() as tape:
            y_pred = model(x)
            # 交叉熵损失
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y_true, y_pred = y_pred) # batch_size个样本的交叉熵损失综合
            loss = tf.reduce_mean(loss) # batch_size个样本的交叉熵损失平均
            print("batch {0}: loss {1}".format(batch_index + 1, loss.numpy()))
            loss_list.append(loss.numpy()) # 记录损失
        if batch_index % 1000 == 0: # 每隔1000个Batch保存一次模型
            path = check_point.save(save_dir + '/mnist_model.ckpt')
            print('model saved to {0}'.format(path))
            # 开启记录器上下文环境
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step = batch_index) # 记录当前损失值

        # 计算损失函数关于权重的梯度
        grads = tape.gradient(loss, model.variables)
        # 根据梯度下降，更新权重
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

    # 绘制损失变化状态
    batch_np = np.arange(0, num_batches)
    loss_np = np.array(loss_list)
    fig, ax = plt.subplots()
    ax.plot(batch_np, loss_np)
    ax.set(xlabel='num_step', ylabel='loss',
       title='loss state')
    ax.grid()
    # plt.show()
    plt.savefig('Handwriting_LOSS_{0}.png'.format(num_batches))

    return model


def evaluation(batch_size, model):
    '''
    @description: 模型评估
    @param 
        batch_size 
        model 训练模型
    @return None
    '''     

    # # 模型评估
    # '''
    #     这里使用tf.keras.metrics中的 SparseCategoricalAccuracy 评估器来评估模型在测试集上的性能。
    #     过程：
    #         1.迭代测试集数据，利用模型预测估计值，然后调用update_state(y_pred, y_true)，来计算预测值和真实值的误差。
    #         2.迭代结束后，调用result()输出最终的评估指标值（预测正确的样本占总样本的比例）
    # '''

    # 实例化模型评估器
    sparse_categorical_accuracy =  tf.keras.metrics.SparseCategoricalAccuracy()

    num_batches_test = int(dataLoader.num_test_data // batch_size) # 计算迭代次数：每次取batch_size个测试集数据

    for batch_index in range(num_batches_test):
        # 计算测试集样本的切片
        start_index, end_index = batch_index * batch_size, (batch_index + 1) *batch_size
        # 计算模型预测值
        y_pred = model.predict(dataLoader.test_data[start_index: end_index]) # [batch_size, 10]
        # print(y_pred, y_pred.shape, type(y_pred))
        y_true = dataLoader.test_label[start_index: end_index]
        # print(y_true, y_true.shape, type(y_true))
        # 评估器进行评估
        sparse_categorical_accuracy.update_state(y_true= y_true, y_pred = y_pred)
    # 输出评估的结构
    print("test accuracy: {0}".format(sparse_categorical_accuracy.result()))
    '''
        test accuracy: 0.9894000291824341
    '''


if __name__ == "__main__":

    starttime = datetime.datetime.now()

    # 数据加载和模型实例化
    dataLoader = MNISTLoader()

    # 模型超参数
    '''
        变量含义可参考：https://blog.csdn.net/m0_37871195/article/details/79829488
    '''
    num_epochs = 5 # 训练的轮数，5轮训练
    batch_size = 50 # 每次迭代训练的数据个数，每次迭代训练50个数据
    num_batches = int(dataLoader.num_train_data // batch_size * num_epochs) # 共60000数据，每次迭代50个数据，训练5轮。所需要的迭代次数 => 6000
    learning_rate = 0.001 # 学习率
    loss_list = [] # 记录迭代损失
    log_dir = './tensorboard/byComplex'
    save_dir = './save/byComplex'

    # 网络训练
    if len(os.listdir(save_dir)) == 0: # 若无训练模型，则训练
        model = train(num_epochs, batch_size, num_batches, learning_rate, log_dir, save_dir)
    else: # 加载训练模型
        model = CNN()
         # 实例化Checkpoint, 指定恢复对象为model
        check_point = tf.train.Checkpoint(myMode = model)
        check_point.restore(tf.train.latest_checkpoint(save_dir))


    # 网络评估
    evaluation(batch_size, model)
    
    endtime = datetime.datetime.now()
    print ('耗费时间：{0}s'.format((endtime - starttime).seconds)) # CPU 102s GPU 31s