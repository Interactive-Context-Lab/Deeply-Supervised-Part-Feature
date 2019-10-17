import os
import tensorflow as tf
from PIL import Image
import numpy as np

dataset_path = 'D:/Projects/ReID_shan/dataset/'
tf_path = dataset_path + 'datas/tfrecords/'
# dataset_name = 'DukeMTMC-reID/'
# train_all_path = dataset_path + dataset_name +'rearrange/train_all/'
# train_path = dataset_path + dataset_name +'rearrange/train/'
# val_path = dataset_path + dataset_name +'rearrange/val/'
# gallery_path = dataset_path + dataset_name +'rearrange/gallery/'
# qurey_path = dataset_path + dataset_name +'rearrange/query/'
all_train = dataset_path +'all/train/'
all_val = dataset_path + 'all/val/'

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'cam_num': tf.FixedLenFeature([], tf.int64),
                                           'real_id': tf.FixedLenFeature([], tf.int64),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [384, 128, 3])  # height, width, img_channel

    label = tf.cast(features['label'], tf.int64)

    cam_num = tf.cast(features['cam_num'], tf.int64)
    real_id = tf.cast(features['real_id'], tf.int64)

    img_batch, label_batch,cam_num_batch,real_id_batch = tf.train.batch([img, label,cam_num,real_id],
                                                                        batch_size=batch_size,
                                                                        capacity=32,
                                                                        allow_smaller_final_batch=True)
    return img_batch, label_batch,cam_num_batch,real_id_batch

def read_and_decode2(filename, batch_size):
    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer([filename], shuffle=False)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'cam_num': tf.FixedLenFeature([], tf.int64),
                                               'real_id': tf.FixedLenFeature([], tf.int64),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [384, 128, 3])  # height, width, img_channel

        label = tf.cast(features['label'], tf.int64)

        cam_num = tf.cast(features['cam_num'], tf.int64)
        real_id = tf.cast(features['real_id'], tf.int64)

        img_batch, label_batch,cam_num_batch,real_id_batch = tf.train.shuffle_batch([img, label,cam_num,real_id],
                                                                                    batch_size=batch_size,
                                                                                    capacity= 1000 + 3 * batch_size,
                                                                                    min_after_dequeue=1000,
                                                                                    allow_smaller_final_batch=True)
        return img_batch, label_batch,cam_num_batch,real_id_batch


def read_and_decode_dq(filename):
    # query
    # with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'cam_num': tf.FixedLenFeature([], tf.int64),
                                           'real_id': tf.FixedLenFeature([], tf.int64),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [384, 128, 3])  # height, width, img_channel

    img_num = tf.cast(features['label'], tf.int64)

    cam_num = tf.cast(features['cam_num'], tf.int64)
    real_id = tf.cast(features['real_id'], tf.int64)

    img_batch, img_num_batch, cam_num_batch, real_id_batch = tf.train.batch([img, img_num, cam_num, real_id],
                                                                            batch_size=1,
                                                                            capacity=32,
                                                                            allow_smaller_final_batch=True)
    return img_batch, img_num_batch, cam_num_batch, real_id_batch

if __name__ == '__main__':

    records_file = "../../datas/tfrecords/train.tfrecords"
    with tf.device('/cpu:0'):
        # img_batch_train, label_batch_train, cam_num_batch,real_id_batch = read_and_decode2(records_file, 32)
        dquery_x, dquery_n, dquery_c, dquery_id = read_and_decode_dq("../../datas/tfrecords_v2/gallery_market.tfrecords")

        # gallery data
        # dgallery_x, dgallery_n, dgallery_c, dgallery_id = read_and_decode2("../../datas/tfrecords/duke_gallery.tfrecords", 32)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for e in range(2):

            # train_samples = 34603
            for i in range(int(15170/16)+1):
                # image_train, label_train, cam_num_train, real_id_train  = sess.run([img_batch_train,label_batch_train,cam_num_batch,real_id_batch])
                image_train, label_train, cam_num_train, real_id_train = sess.run(
                    [dquery_x, dquery_n, dquery_c, dquery_id])
                print(i)
                print(image_train.shape)
                print(label_train.shape)
                print(cam_num_train)
                print(real_id_train)
                for j in range(32):
                    pass
                    # img = Image.fromarray(image_train[j], 'RGB')  # 这里Image是之前提到的
                    # img.save('C:/Users/Rachel/Desktop/test/' + str(i) + '_'+str(j)+'Label_' + str(label_train[j])+'-c'+str(cam_num_train[j])+str(real_id_train[j]) + '.jpg')  # 存下图片
                    # img.save('C:/Users/Rachel/Desktop/test2/' +  'Label_' + str(
                    #     label_train[j]) +'_'+str(cam_num_train[j])+ '_'+str(e) + '_'+str(i) + '_' + str(j) + '.jpg')  # 存下图片

        coord.request_stop()
        coord.join(threads)