import os
from PIL import Image
import tensorflow as tf

dataset = 'duke'
dataset_path = '../../datas/'
dataset_name = 'images/rearrange/duke/'
tf_path = dataset_path + 'tfrecords/'  # output path
train_all_path = dataset_path + dataset_name
train_path = dataset_path + dataset_name +'bounding_box_train/'  # dirname in dataset for training set
gallery_path = dataset_path + dataset_name +'bounding_box_test/' # dirname in dataset for gallery
qurey_path = dataset_path + dataset_name +'query/'               # dirname in dataset for query

def create_record(path, name, dataset):
    print('start')
    classes = os.listdir(path)
    print('classes: ', len(classes))

    writer = tf.python_io.TFRecordWriter(tf_path+'%s_%s.tfrecords' % name, dataset)
    for index, name in enumerate(classes):
        class_path = path + name + "/"
        for img_name in os.listdir(class_path):
            print(img_name)
            img_name_split = img_name.split('_')
            real_id = int(img_name_split[0])
            cam_num = int(img_name_split[1])
            # print('cam_num:', cam_num)
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((128, 384)) # width, height
            # print(np.shape(img))
            img_raw = img.tobytes() #将图片转化为原生bytes

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'cam_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[cam_num])),
                'real_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[real_id]))
            }))
            writer.write(example.SerializeToString())
    print('Finish')
    writer.close()

if __name__ == '__main__':
    create_record(train_path, "train", dataset)
    create_record(gallery_path, "gallery", dataset)
    create_record(qurey_path, "query", dataset)

