import tensorflow as tf

######################################################################
# Reading tfrecords file
# --------
def read_and_decode_t(filename, batch_size):
    # training
    # with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)

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

    img_batch, label_batch, cam_batch, id_batch = tf.train.shuffle_batch([img, label, cam_num, real_id], batch_size=batch_size,
                                                    capacity=1000 + 32 * batch_size,
                                                    min_after_dequeue=1000,
                                                    allow_smaller_final_batch=True)

    return [img_batch, label_batch, cam_batch, id_batch]


def read_and_decode_mq(filename):
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


def read_and_decode_mg(filename):
    # gallery
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


def read_and_decode_dg(filename):
    # gallery
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