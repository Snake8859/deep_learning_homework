'''
Author: your name
Date: 2021-04-21 11:40:31
LastEditTime: 2021-06-02 14:39:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \code\data_loader.py
'''
import os
import tensorflow as tf

class CatDogLoader():
    def __init__(self, file_list, buffer_size, batch_size, num_epoch):
        # print(file_list[0:10])
        self.image_data = {}
        # 遍历文件
        for file_path in file_list:
            file_root, file_name = os.path.split(file_path)
            label = file_name.split('.')[0]
            label_name = None
            # 根据文件名打标签
            if(label == 'cat'):
                label_name = 0
            elif(label == 'dog'):
                label_name = 1
            
            if(label_name != None):
                self.image_data[file_path] = label_name
        # print(self.image_data)

        all_image_path = list(self.image_data)
        all_image_label = list(self.image_data.values())
        ds = tf.data.Dataset.from_tensor_slices((all_image_path, all_image_label))

        image_label_ds = ds.map(self.load_and_preprocess_image)
        # buffer_size（随机缓冲区大小）:设置一个和数据集大小一致的 shuffle, 以保证数据被充分打乱
        image_label_ds = image_label_ds.shuffle(buffer_size = buffer_size)
        # batch:数据打包分组,每batch_size个分数据成一组
        image_label_ds = image_label_ds.batch(batch_size)
        # count:数据重复多少epoch, 训练的轮数
        self.image_label_ds = image_label_ds.repeat(count = num_epoch)


    def load_and_preprocess_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels = 3)  # 编码图像
        image = tf.image.resize(image, (224, 224)) # 图像统一尺寸
        image /= 255.0 # 图像归一化到[0-1]
        return image, label

            


# 定义Feature结构，告诉解码器每个Feature的类型是什么，类似于一个数据集的"描述文件“
feature_desciption = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_example(example_string):
    '''
        对TFRecord文件中的每一个序列化文件的tf.train.Example解码
    '''
    feature_dict = tf.io.parse_single_example(example_string, feature_desciption)
    # print(feature_dict)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image']) # 解码JPGE图像
    image_resized = tf.image.resize(feature_dict['image'], [299, 299]) / 255
    return image_resized, feature_dict['label']


def read_tfrecord(tfrecord_dir):
    '''
        读取TFRecord文件
    '''
    raw_dataset = tf.data.TFRecordDataset(tfrecord_dir) # 读取TFRecord文件
    dataset = raw_dataset.map(_parse_example)
    
    # for image, label in dataset:
    #     print(image, image.shape, type(image))
    #     # print(label, label.shape, type(label))
    #     plt.title('cat' if label == 0 else 'dog')
    #     plt.imshow(image.numpy())
    #     plt.show()
    #     break

    return dataset
        
    