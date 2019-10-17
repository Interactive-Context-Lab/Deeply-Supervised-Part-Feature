import argparse
import os, sys
import shutil
import csv
import time
import numpy as np
# np.set_printoptions(threshold=np.inf)

from utils.utils import ReDirectSTD
from utils.re_ranking import re_ranking
from utils.metric import compute_score
from utils.distance import compute_dist, normalize
from utils.read_decode import *
from utils.data_aug import data_augmentation

# from net.PartNetwork import Partbased
# from net.se_resnext_backbone import SE_ResNeXt
from model import DSPF
import tensorflow.contrib.slim as slim
import tensorflow as tf

######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
# parser.add_argument('--record_dir', type=str, default='E:/datasets/openset/TTR_FTR_10fold/market/test9/tfrecords/')
parser.add_argument('--record_dir', type=str, default='D:/datas/tfrecords_v3/')
parser.add_argument('--feature_dir', type=str, default='features/')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--dataset', type=str, default='market', choices=['market', 'duke', 'all'])
parser.add_argument('--device', type=int, default=1, choices=[0, 1])
parser.add_argument('--num_dims', type=int, default=256)

# train parameter
parser.add_argument('--reuse', type=bool, default=False)
parser.add_argument('--reuse_path', type=str, default='model/190828_041556/')
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--pre_model', type=str, default='pretrain/se_resnext50/se_resnext50.ckpt')
# parser.add_argument('--pre_model', type=str, default='model/190827_123027/model.ckpt-100')
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001) #0.001
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--drop_rate', type=float, default=0.2)

parser.add_argument('--lamda', type=float, default=0.8)
parser.add_argument('--data_aug', type=bool, default=True)

# Part based parameter
parser.add_argument('--attention', type=bool, default=True)
parser.add_argument('--deeplysupervised', type=bool, default=True)

parser.add_argument('--part', type=bool, default=True)
parser.add_argument('--all', type=bool, default=False)
# parser.add_argument('--half', type=bool, default=False)
# parser.add_argument('--straight', type=bool, default=False)


# test parameter
parser.add_argument('--only_test', type=bool, default=False)
parser.add_argument('--logfile', type=bool, default=True)
parser.add_argument('--test_one', type=bool, default=False)
parser.add_argument('--restore_model', type=str,
                    default='model/190711_130425_market_test9_2/model.ckpt-95') # set required if test one is True
parser.add_argument('--ckpt_path', type=str, default='model/190828_041556')
parser.add_argument('--skip_model', type=int, default=0)
parser.add_argument('--feat_ext', type=bool, default=True)
parser.add_argument('--normalize_feat', type=bool, default=True)
parser.add_argument('--rerank', type=bool, default=True)

######################################################################
# Settings
# --------
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)

train_file = args.record_dir + 'train_' + args.dataset + '.tfrecords'
q_file = args.record_dir + 'query_' + args.dataset + '.tfrecords'
g_file = args.record_dir + 'gallery_' + args.dataset + '.tfrecords'

class_num = 709  # duke 702 #market 751
                 # if open-set: duke: 709, market: 758
if args.dataset == 'market':
    class_num = 751
# train_samples = 12936  #duke: 16552 # market: 12936
train_samples = sum([1 for _ in tf.python_io.tf_record_iterator(train_file)])
query_samples = sum([1 for _ in tf.python_io.tf_record_iterator(q_file)])
gallery_samples = sum([1 for _ in tf.python_io.tf_record_iterator(g_file)])
print(train_samples, query_samples, gallery_samples)
# mq_samples = 3368
# mg_samples = 12547
# dq_samples = 2228
# dg_samples = 15631
iteration = int(train_samples / args.batch_size + 1)
g_testfeat = args.feature_dir + 'gallery.csv'
q_testfeat = args.feature_dir + 'query.csv'

if not os.path.exists(os.path.dirname(g_testfeat)):
    os.makedirs(os.path.dirname(g_testfeat))

# training data
###################
# Loading Dataset #
###################
with tf.device("/cpu:0"):
    train_x, train_y, _, _ = read_and_decode_t(train_file, args.batch_size)

    # query data
    query_x, query_n, query_c, query_id = read_and_decode_mq(q_file)
    # gallery data
    gallery_x, gallery_n, gallery_c, gallery_id = read_and_decode_mg(g_file)


if args.data_aug:
    train_x = tf.cast(train_x, dtype=tf.float32)
    train_x = tf.image.random_saturation(train_x, lower=0.5, upper=1.5)
    train_x = tf.image.random_contrast(train_x, lower=0.5, upper=1.5)
    train_x = tf.image.random_brightness(train_x, max_delta=0.4)


#################
# Model Setting #
#################
if not args.only_test:
    x = tf.placeholder(tf.float32, shape=[args.batch_size, 384, 128, 3])  # input image size
    label = tf.placeholder(tf.int64, shape=[args.batch_size])
else:
    x = tf.placeholder(tf.float32, shape=[1, 384, 128, 3])  # input image size

training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

#############   get model   ###############################
DSPF_model = DSPF(x=x, class_num=class_num, training_flag=training_flag, num_dims=args.num_dims, drop_rate=args.drop_rate,
             atten=args.attention, ds=args.deeplysupervised, part=args.part, all=args.all)

end_points, deeplysupervised_list, logits_list, local_x_list = DSPF_model.get_endpoints()

######################################################################
# Testing Functions
# --------
def feature_extract(sess, q_featfile, g_featfile):
    '''
    Extracting features of all images to save as csv file
    :param sess: tensorflow's session object 
    :param q_featfile: query csv file saving path
    :param g_featfile: gallery csv file saving path
    :return: None
    '''
    f = open(q_featfile, "w", newline='')
    w = csv.writer(f)
    f2 = open(g_featfile, "w", newline='')
    w2 = csv.writer(f2)

    a = ['img_num', 'real_id', 'camera', 'features']
    w.writerow(a)
    w2.writerow(a)

    print('Exracting feature...')
    print('extract {}'.format(args.dataset))
    extract_st = time.time()
    last_t = time.time()
    for step in range(query_samples):
        query_batch_x, query_batch_n, query_batch_c, query_batch_id = sess.run(
            [query_x, query_n, query_c, query_id])

        test_feed_dict = {
            x: query_batch_x,
            training_flag: False
        }
        features = sess.run([local_x_list], feed_dict=test_feed_dict)
        ## [12, 256]
        flen = int(np.shape(features)[1])
        b = np.reshape(features, (flen, 256))
        c = np.reshape(b, flen * 256)
        d = np.array([query_batch_n, query_batch_id, query_batch_c])
        q_out = np.append(d, c)
        w.writerow(q_out)

        if step % 300 == 0:
            print('Train {}/{} done, +{:.2f}s, total {:.2f}s'.format(
                step, query_samples, time.time() - last_t, time.time() - extract_st))
            last_t = time.time()
    f.close()
    print('Done, {} samples, {:.2f}s, {} FPS\n'.format(
        query_samples, time.time() - extract_st, query_samples / (time.time() - extract_st)))

    extract_st2 = time.time()
    last_t2 = time.time()
    for step in range(gallery_samples):
        gallery_batch_x, gallery_batch_n, gallery_batch_c, gallery_batch_id = sess.run\
            ([gallery_x, gallery_n, gallery_c, gallery_id])

        test_feed_dict = {
            x: gallery_batch_x,
            training_flag: False
        }
        features = sess.run([local_x_list], feed_dict=test_feed_dict)
        flen = int(np.shape(features)[1])
        b = np.reshape(features, (flen, 256))
        c = np.reshape(b, flen * 256)
        d = np.array([gallery_batch_n, gallery_batch_id, gallery_batch_c])
        q_out = np.append(d, c)
        w2.writerow(q_out)

        if step % 1300 == 0:
            print('Gallery {}/{} done, +{:.2f}s, total {:.2f}s'.format(
                step, gallery_samples, time.time() - last_t2, time.time() - extract_st2))
            last_t2 = time.time()
    f2.close()
    print('Done, {} samples, {:.2f}s, {} FPS\n'.format(
        gallery_samples, time.time() - extract_st2, gallery_samples / (time.time() - extract_st2)))

def test(sess, rerank=True):
    def print_scores(mAP, cmc_scores):
        print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'.format(mAP, *cmc_scores[[0, 4, 9]]))

    q_featfile = q_testfeat
    g_featfile = g_testfeat

    if args.feat_ext:
        feature_extract(sess, q_featfile, g_featfile)

    q_info = np.genfromtxt(q_featfile, delimiter=',', skip_header=1)
    g_info = np.genfromtxt(g_featfile, delimiter=',', skip_header=1)
    print('shape of q_info2: ', np.shape(q_info[:, 3:]))
    print('shape of q_info2: ', np.shape(g_info[:, 3:]))

    if args.normalize_feat:
        q_feat = normalize(q_info[:, 3:], axis=1)
        g_feat = normalize(g_info[:, 3:], axis=1)
    else:
        q_feat = q_info[:, 3:]
        g_feat = g_info[:, 3:]


    ################
    # Single Query #
    ################
    # query-gallery distance
    print('Computing distance...')
    st = time.time()
    q_g_dist = compute_dist(q_feat, g_feat, type='euclidean')
    print('Done, {:.2f}s, shape of q_g_dist: {}'.format(time.time() - st, np.shape(q_g_dist)))

    print('Computing scores...')
    st = time.time()
    mAP, cmc_scores = compute_score(q_g_dist,
                                    query_ids=q_info[:, 1],
                                    gallery_ids=g_info[:, 1],
                                    query_cams=q_info[:, 2],
                                    gallery_cams=g_info[:, 2])
    print('Done, {:.2f}s'.format(time.time() - st))

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)

    if not rerank:
        return mAP, cmc_scores

    ##########################
    # Re-ranked Single Query #
    ##########################
    print('Computing Re-ranking distance...')
    st = time.time()
    # query-query distance
    q_q_dist = compute_dist(q_feat, q_feat, type='euclidean')
    print('shape of q_q_dist: ', np.shape(q_q_dist))

    # gallery-gallery distance
    g_g_dist = compute_dist(g_feat, g_feat, type='euclidean')
    print('shape of g_g_dist: ', np.shape(g_g_dist))

    # re-ranked query-gallery distance
    re_r_q_g_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    print('shape of re_r_q_g_dist: ', np.shape(re_r_q_g_dist))
    print('Done, {:.2f}s'.format(time.time() - st))

    print('Computing scores for re-ranked distance...')
    st = time.time()
    remAP, recmc_scores = compute_score(re_r_q_g_dist,
                                        query_ids=q_info[:, 1],
                                        gallery_ids=g_info[:, 1],
                                        query_cams=q_info[:, 2],
                                        gallery_cams=g_info[:, 2])

    print('Done, {:.2f}s'.format(time.time() - st))

    print('{:<30}'.format('Re-ranked Single Query:'), end='')
    print_scores(remAP, recmc_scores)

    return mAP, cmc_scores, remAP, recmc_scores

def test_one():
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('\n\nLoading model......')
        st = time.time()
        print('Restore from ', args.restore_model)
        saver.restore(sess, args.restore_model)
        print('Finished loading, {:.2f}s\n'.format(time.time() - st))

        test(sess, rerank=args.rerank)

        coord.request_stop()
        coord.join(threads)
    return True

def test_all():
    def print_best(idx, mAP, cmc_scores, remAP, recmc_scores):
        print("\nThe best:")
        print("model: {}".format(idx))
        print('Inital: [mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'.format(mAP, *cmc_scores[[0, 4, 9]]))
        print('Reranking: [mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]\n'.format(remAP, *recmc_scores[[0, 4, 9]]))

    saver = tf.train.Saver()
    ckpt_list = tf.train.get_checkpoint_state(args.ckpt_path)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        test_writer = tf.summary.FileWriter(folder_now + '/logs/test', sess.graph)

        best_mAP, best_cmc_scores, best_remAP, best_recmc_scores = 0.0, 0.0, 0.0, 0.0
        best_model_idx = 0
        for i, ckpt in enumerate(ckpt_list.all_model_checkpoint_paths):
            if i + 1 < args.skip_model: continue
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            print('\n\nLoading model......')
            st = time.time()
            print('Restore from ', ckpt)
            saver.restore(sess, ckpt)
            print('Finished loading, {:.2f}s\n'.format(time.time() - st))

            mAP, cmc_scores, remAP, recmc_scores = test(sess, rerank=args.rerank)
            test_summary = tf.Summary(value=[tf.Summary.Value(tag='mAP', simple_value=mAP, ),
                                            tf.Summary.Value(tag='Reranking_mAP', simple_value=remAP, ),
                                            tf.Summary.Value(tag='CMC1', simple_value=cmc_scores[0]),
                                            tf.Summary.Value(tag='Reranking_CMC1', simple_value=recmc_scores[0])])

            test_writer.add_summary(summary=test_summary, global_step=i+1)
            test_writer.flush()

            if float(best_mAP) < float(mAP):
                best_mAP, best_cmc_scores, best_remAP, best_recmc_scores = \
                mAP, cmc_scores, remAP, recmc_scores
                best_model_idx = i + 1

            print_best(best_model_idx, best_mAP, best_cmc_scores, best_remAP, best_recmc_scores)
        test_writer.close()
        coord.request_stop()
        coord.join(threads)
    return 1

######################################################################
# training Functions
# --------
def train():
    loss_list = []
    dsloss_list = []
    accuracy_list = []

    for ds in deeplysupervised_list:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=ds))
        dsloss_list.append(loss)
        correct_prediction_p = tf.equal(tf.argmax(ds, 1), label)
        accuracy_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))
        accuracy_list.append(accuracy_p)

    for logits in logits_list:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
        loss_list.append(loss)
        correct_prediction_p = tf.equal(tf.argmax(logits, 1), label)
        accuracy_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))
        accuracy_list.append(accuracy_p)

    cost = tf.reduce_sum(loss_list)
    ds_loss = tf.reduce_sum(dsloss_list)

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = cost * args.lamda + ds_loss * (1.0 - args.lamda) + l2_loss * args.weight_decay

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = optimizer.minimize(total_loss)

    accuracy = tf.reduce_mean(tf.cast(accuracy_list, tf.float32))

    # training start
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    # variables = slim.get_variables_to_restore()

    # Creating a restorer that only restore weights of backbone from pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    base_list = [v for v in vars_list if
                 "conv1" in v.name.split("/")[0].split("_")[0] or
                 "conv2" in v.name.split("/")[0].split("_")[0] or
                 "conv3" in v.name.split("/")[0].split("_")[0] or
                 "conv4" in v.name.split("/")[0].split("_")[0] or
                 "conv5" in v.name.split("/")[0].split("_")[0]]
    # variables_to_restore2 = [v for v in variables
    #                         if "is_training" not in v.name
    #                         and "beta1_power" not in v.name
    #                         and "beta2_power" not in v.name
    #                         and "Adam" not in v.name
    #                         and 'all' not in v.name
    #                         # and 'half' not in v.name
    #                         # and 'straight' not in v.name
    #                         and 'part' not in v.name
    #                         # and 'deeplysupervised' not in v.name
    #                         and 'Attention' not in v.name
    #                         ]
    pretrain_restorer = tf.train.Saver(base_list)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        print('Initialing all variables......')
        sess.run(tf.global_variables_initializer())
        print('Finished initialing.')
        start_epoch = 0
        if args.reuse:
            lastest_ckpt = tf.train.get_checkpoint_state(args.reuse_path).model_checkpoint_path  #.model_checkpoint_path
            start_epoch = int(lastest_ckpt.split('-')[1])
            saver.restore(sess, lastest_ckpt)
        # -----plot the figures on tensorboard------
        train_writer = tf.summary.FileWriter(folder_now + '/logs/train', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        if (not args.reuse) and args.pre_model != '':
            print("Using Pretrain model")
            print('Start loading......')
            st = time.time()
            print('Restore from ', args.pre_model)
            if args.pre_model == 'pretrain/se_resnext50/se_resnext50.ckpt':
                pretrain_restorer.restore(sess, args.pre_model)
            else:
                saver.restore(sess, args.pre_model)

            print('Finished loading, {:.2f}s\n'.format(time.time() - st))

        #############
        # Training #
        ############
        print('Start Training......')
        train_st = time.time()
        total_iter = 0
        epoch_learning_rate = args.learning_rate
        for epoch in range(1, args.total_epochs + 1):
            if epoch <= start_epoch: continue
            epoch_st = time.time()
            print('Epoch: ', epoch)

            if epoch == 60:
                epoch_learning_rate = epoch_learning_rate / 10.0

            train_acc = 0.0
            train_loss = 0.0

            step_st = time.time()
            for step in range(1, iteration + 1):
                batch_x, batch_y = sess.run([train_x, train_y])
                if args.data_aug:
                    batch_x = data_augmentation(batch_x)  # flip images and translation and erasing

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss, batch_acc = \
                    sess.run([train, total_loss, accuracy], feed_dict=train_feed_dict)

                total_iter += 1
                if step % 200 == 0:
                    print('Ep {}, Iter {}, Step {}/{}, {:.2f}s, batch_loss: {:.4f}, batch_acc: {:.4f}'.format(
                        epoch, total_iter, step, iteration, time.time() - step_st, batch_loss, batch_acc))
                    step_st = time.time()
                train_loss += batch_loss
                train_acc += batch_acc

            train_loss /= iteration  # average loss
            train_acc /= iteration  # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='avg_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='avg_accuracy', simple_value=train_acc)])

            print('Epoch training time {:.2f}s, Total training time {:.2f}s'.format(
                time.time() - epoch_st, time.time() - train_st))

            train_writer.add_summary(summary=train_summary, global_step=epoch)
            train_writer.flush()

            print('epoch: {}/{}, lr: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
                epoch, args.total_epochs, epoch_learning_rate, train_loss, train_acc))

            print('\nSaving model to ', folder_now + '\n')
            saver.save(sess=sess, save_path=folder_now + '/model.ckpt', global_step=epoch)

        coord.request_stop()
        coord.join(threads)

######################################################################
# Main
# --------

if __name__ == "__main__":
    global folder_now

    if args.only_test:
        folder_now = args.ckpt_path
        if args.logfile:
            file = time.strftime("%y%m%d_%H%M%S", time.localtime()) + 'stdout.txt'
            ReDirectSTD(folder_now + '/' + file, 'stdout', True)  # logging output information into txtfile

        print('Used test dataset:', q_file)
        if args.test_one:
            test_one()
        else:
            test_all()
        sys.exit("Finish testing")

    else:
        # 複製檔案作為紀錄 > 當前檔案, 目標路徑
        if args.reuse:
            folder_now = args.reuse_path
        else:
            folder_now = args.model_dir + time.strftime("%y%m%d_%H%M%S", time.localtime())
            setting_foder = folder_now + "/settings/"
            if not os.path.exists(folder_now):
                os.makedirs(folder_now)
                os.makedirs(setting_foder)
            ReDirectSTD(folder_now + '/train_stdout.txt', 'stdout', True)  # logging output information into txtfile
            shutil.copyfile(__file__, setting_foder + __file__.split("/")[-1])
            shutil.copyfile(os.path.dirname(__file__) + "/model.py", setting_foder + "model.py")
            shutil.copyfile(os.path.dirname(__file__) + "/net/PartNetwork.py", setting_foder + "PartNetwork.py")
            shutil.copyfile(os.path.dirname(__file__) + "/net/se_resnext_backbone.py", setting_foder + "se_resnext_backbone.py")

        print('Used train dataset:', train_file)
        train()

