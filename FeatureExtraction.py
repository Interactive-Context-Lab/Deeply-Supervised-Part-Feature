import argparse
import os
import csv
import time
import numpy as np
# np.set_printoptions(threshold=np.inf)

from utils.read_decode import read_and_decode
from model import DSPF

import tensorflow as tf
######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
# parser.add_argument('--class_num', type=int, default=702)
parser.add_argument('--dataset', type=str, default='duke', choices=['market', 'duke', 'all', 'EE3f'])
parser.add_argument('--record_dir', type=str, default='E:/datasets/openset/TTR_FTR/')


parser.add_argument('--feature_dir', type=str, default='features/duke_1part_atten_multistage/')
parser.add_argument('--restore_model', type=str,
                    default='D:/reid_approach/model/190730_124641_duke_1part_atten_multistage/model.ckpt-100')


parser.add_argument('--device', type=int, default=1, choices=[0, 1])
parser.add_argument('--batch_size', type=int, default=1)  # can only set to 1
parser.add_argument('--iteration', type=int, default=10)  # the testing has repeat for 10 time
parser.add_argument('--extract_gallery', type=bool, default=True)
parser.add_argument('--extract_query', type=bool, default=True)

# Part based parameter
parser.add_argument('--attention', type=bool, default=True)

parser.add_argument('--part', type=bool, default=False)
parser.add_argument('--all', type=bool, default=True)
# parser.add_argument('--half', type=bool, default=False)
# parser.add_argument('--straight', type=bool, default=False)


parser.add_argument('--drop_rate', type=float, default=0.2)

######################################################################
# Settings
# --------
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)

args.class_num = 702  # duke 702 #market 751
                 # if open-set: duke: 709, market: 758
if args.dataset == 'market':
    args.class_num = 751

x = tf.placeholder(tf.float32, shape=[None, 384, 128, 3])  # input image size
training_flag = tf.placeholder(tf.bool)

#############   get model   ###############################
DSPF_model = DSPF(x=x, class_num=args.class_num, training_flag=training_flag, num_dims=args.num_dims, drop_rate=args.drop_rate,
             atten=args.attention, ds=args.deeplysupervised, part=args.part, all=args.all)
_, _, _, local_x_list = DSPF_model.get_endpoints()
###########################################################


def feature_extract(sess, featfile, num_sample, batch_x, batch_n, batch_c, batch_id):
    f = open(featfile, "w", newline='')
    w = csv.writer(f)
    a = ['img_num', 'real_id', 'camera', 'features']
    w.writerow(a)

    print('Exracting feature...')
    extract_st = time.time()
    last_t = time.time()
    for step in range(num_sample):
        np_batch_x, np_batch_n, np_batch_c, np_batch_id = sess.run(
            [batch_x, batch_n, batch_c, batch_id])
        test_feed_dict = {
            x: np_batch_x,
            training_flag: False
        }
        features = sess.run([local_x_list], feed_dict=test_feed_dict)
        flen = int(np.shape(features)[1])
        b = np.reshape(features, (flen, 256))
        c = np.reshape(b, flen * 256)
        d = np.array([np_batch_n, np_batch_id, np_batch_c])
        q_out = np.append(d, c)
        w.writerow(q_out)
        f.flush()
        if step % 200 == 0:
            print('features {}/{} done, +{:.2f}s, total {:.2f}s'.format(
                step, num_sample, time.time() - last_t, time.time() - extract_st))
            last_t = time.time()
    f.close()
    print('Done, {} samples, {:.2f}s, {} FPS\n'.format(
        num_sample, time.time() - extract_st, num_sample / (time.time() - extract_st)))

####  main function  #####
def run_test():
    tfrecordpath = "{}/{}/test{}/tfrecords/{}_{}.tfrecords"
    features_out = "{}/{}/test{}/{}.csv"
    saver = tf.train.Saver()
    if args.extract_gallery:
        for i in range(0, args.iteration):
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print('\n\nLoading model......')
                st = time.time()
                print('Restore from ', args.restore_model)
                saver.restore(sess, args.restore_model)
                print('Finished loading, {:.2f}s\n'.format(time.time() - st))

                tffilepath = tfrecordpath.format(args.record_dir, args.dataset, i, "gallery", args.dataset)
                testfeat = features_out.format(args.feature_dir, args.dataset, i, "gallery")
                if not os.path.exists(os.path.dirname(testfeat)):
                    os.makedirs(os.path.dirname(testfeat))
                print("Extraction: %s, write: %s" % (tffilepath, testfeat))

                num_sample = sum([1 for _ in tf.python_io.tf_record_iterator(tffilepath)])
                batch_x, batch_n, batch_c, batch_id = read_and_decode(tffilepath, args.batch_size)


                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                feature_extract(sess, testfeat, num_sample, batch_x, batch_n, batch_c, batch_id)

                coord.request_stop()
                coord.join(threads)

    if args.extract_query:
        for i in range(0, args.iteration):
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print('\n\nLoading model......')
                st = time.time()
                print('Restore from ', args.restore_model)
                saver.restore(sess, args.restore_model)
                print('Finished loading, {:.2f}s\n'.format(time.time() - st))
                tffilepath = tfrecordpath.format(args.record_dir, args.dataset, i, "query", args.dataset)
                testfeat = features_out.format(args.feature_dir, args.dataset, i, "query")
                if not os.path.exists(os.path.dirname(testfeat)):
                    os.makedirs(os.path.dirname(testfeat))
                print("Extraction: %s, write: %s" % (tffilepath, testfeat))
                num_sample = sum([1 for _ in tf.python_io.tf_record_iterator(tffilepath)])

                batch_x, batch_n, batch_c, batch_id = read_and_decode(tffilepath)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                feature_extract(sess, testfeat, num_sample, batch_x, batch_n, batch_c, batch_id)

                coord.request_stop()
                coord.join(threads)

if __name__ == "__main__":
    run_test()