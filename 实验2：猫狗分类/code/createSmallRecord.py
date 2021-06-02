import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
    TFRecord是TensorFlow中的数据集存储格式，可以高效地读取和处理数据集

    TFRecord可以理解为一系列序列化的tf.train.Example元素所组成的列表文件，每一个tf.train.Example又由若干个tf.train.Feature的字典组成

    TFRecord创建：
        - 将数据转为tf.train.Feature
        - 将Feature转为Example
        - 将Example序列化为字符串，写入TFRecord文件

    TFRecord读取：
        - 通过tf.Data.TFrecordDataset读取原始的TFRecord文件，得到Dataset对象
        - 通过Dataset.map方法，将数据集每一个序列化的Example对象，通过tf.io.parse_single_example反序列化
'''

def create_tfrecord(root, all_filenames, tfrecord_dir):
    '''
        创建tfrecord文件
    '''
    # 创建标签
    all_labels = []
    for fname in all_filenames:
        if(fname.startswith('cat')):
            all_labels.append(0)
        elif(fname.startswith('dog')):
            all_labels.append(1)   
         
    # TFrecord创建上下文环境
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        # 遍历原始数据
        for filename, label in zip(all_filenames, all_labels):
            # 读取图片
            file_path = os.path.join(root, filename)
            print(file_path, label)
            image = open(file_path, 'rb').read()
            # 创建Feautre
            feature = {
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])), # 图片是Bytes对象
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])) # 标签是Int对象
            }
            # 创建Example
            example = tf.train.Example(features  = tf.train.Features(feature = feature))

            # 序列化写入TFRecord文件
            writer.write(example.SerializeToString())


def read_tfrecord(tfrecord_dir):
    '''
        读取TFRecord文件
    '''
    raw_dataset = tf.data.TFRecordDataset(tfrecord_dir) # 读取TFRecord文件
    dataset = raw_dataset.map(_parse_example).shuffle(3000).batch(16)

    for images, labels in dataset:
        # print(image.shape, type(image)) (16, 299, 299, 3)
        # print(label.shape, type(label)) (16,)
        col = 0
        for i in range(16):
            img = images[i]
            img_label = labels[i]
            # print(img.shape, img_label.shape) # (28, 28, 1)
            cur_spec = (col, i % 4)
            if (i + 1) % 4 == 0: # 每4个换行
                col +=1
            plt.subplot2grid((4, 4), cur_spec)
            plt.imshow(img.numpy(), cmap='gray')
            plt.title('cat' if img_label == 0 else 'dog')
            plt.axis('off')

        plt.show()
        break



def _parse_example(example_string):
    '''
        对TFRecord文件中的每一个序列化文件的tf.train.Example解码
    '''
    feature_dict = tf.io.parse_single_example(example_string, feature_desciption)
    # print(feature_dict)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image']) # 解码JPGE图像
    image_resized = tf.image.resize(feature_dict['image'], [224, 224]) / 255
    return image_resized, feature_dict['label']


if __name__ == "__main__":

    
    # 数据根路径
    data_root = './dogs-vs-cats/'

    # tfrecord保存路径
    train_tfrecord_dir =  os.path.join(data_root, 'small_train.tfrecords')
    test_tfrecord_dir =  os.path.join(data_root, 'small_test.tfrecords')
    valid_tfrecord_dir =  os.path.join(data_root, 'small_valid.tfrecords')


    # 定义Feature结构，告诉解码器每个Feature的类型是什么，类似于一个数据集的"描述文件“
    feature_desciption = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }


    # 创建训练集
    train_dir = os.path.join(data_root, 'train')
    train_all_filenames = os.listdir(train_dir)
    create_tfrecord(train_dir, train_all_filenames, train_tfrecord_dir)
    # read_tfrecord(train_tfrecord_dir)

    # 创建测试集
    test_dir = os.path.join(data_root, 'test')
    test_all_filenames = os.listdir(test_dir)
    create_tfrecord(test_dir, test_all_filenames, test_tfrecord_dir)
    # read_tfrecord(test_tfrecord_dir)

    # 创建验证集
    valid_dir = os.path.join(data_root, 'valid')
    valid_all_filenames = os.listdir(valid_dir)
    create_tfrecord(valid_dir, valid_all_filenames, valid_tfrecord_dir)
    read_tfrecord(valid_tfrecord_dir)
