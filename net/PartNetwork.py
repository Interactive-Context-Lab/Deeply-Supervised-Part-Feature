import tensorflow as tf
from tflearn.layers.conv import avg_pool_2d, max_pool_2d, global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

######################################################################
# Model Definetions
# --------
def conv_layer(input, filter, kernel, stride=1, use_bias=True, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=use_bias, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding, name=layer_name)
        if activation:
            network = tf.nn.relu(network)
        return network

def Fully_connected(x, units, use_bias=True, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=use_bias, units=units, name=layer_name)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers):
    return tf.concat(layers, axis=3)

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = global_avg_pool(input_x, name='Global_avg_pooling')
        # squeeze = input_x
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation

        return scale

def Part_extract(input_x, layer_name, class_num, training, num_dims=256, drop_rate=0.2):
    with tf.name_scope(layer_name):
        output_height = int(np.shape(input_x)[1])
        output_weight = int(np.shape(input_x)[2])
        stripe_w = output_weight
        stripe_h = int(output_height / 6)
        local_x_list = []
        logits_list = []

        for i in range(6):  # 6 parts
            local_x = avg_pool_2d(input_x[:, i * stripe_h: (i + 1) * stripe_h, :, :], (stripe_h, stripe_w))
            local_x = tf.layers.dropout(local_x, rate=drop_rate, training=training,
                                        name=layer_name + '_drop1')
            local_x = conv_layer(local_x, filter=num_dims, kernel=[1, 1], layer_name=layer_name + '_split_conv' + str(i))
            local_x = Batch_Normalization(local_x, training=training, scope=layer_name + '_batch' + str(i))
            local_x = flatten(local_x)
            local_x_list.append(local_x)
            # local_x = tf.layers.dropout(local_x, rate=drop_rate, training=training,
            #                             name=layer_name + '_drop2')
            local_x = Fully_connected(local_x, class_num, use_bias=False,
                                      layer_name=layer_name + '_fully_connected_' + str(i))
            logits_list.append(local_x)

        return local_x_list, logits_list

# 1 part
def All_extract(input_x, layer_name, class_num, training, num_dims=256, drop_rate=0.2):
    with tf.name_scope(layer_name):
        output_height = int(np.shape(input_x)[1])
        output_weight = int(np.shape(input_x)[2])

        local_x_list = []
        logits_list = []

        local_x1 = avg_pool_2d(input_x[:, :, :, :], (output_height, output_weight), name=layer_name + '_AvgPool2D')
        # local_x1 = tf.layers.dropout(local_x1, rate=drop_rate, training=training,
        #                              name=layer_name + '_drop1')
        local_x1 = conv_layer(local_x1, filter=num_dims, kernel=[1, 1], layer_name=layer_name + '_split_conv')
        local_x1 = Batch_Normalization(local_x1, training=training, scope=layer_name + '_batch')
        local_x1 = flatten(local_x1)
        local_x_list.append(local_x1)
        # local_x1 = tf.layers.dropout(local_x1, rate=drop_rate, training=training,
        #                              name=layer_name + '_drop2')
        local_x1 = Fully_connected(local_x1, class_num, use_bias=False, layer_name=layer_name + '_fully_connected')
        logits_list.append(local_x1)

        return local_x_list, logits_list

# horizontal 2 parts
def Half_extract(input_x, layer_name, class_num, training, drop_rate=0.2):
    with tf.name_scope(layer_name):
        output_height = int(np.shape(input_x)[1])
        output_weight = int(np.shape(input_x)[2])
        stripe_w2 = output_weight
        stripe_h2 = int(output_height / 2)
        local_x_list = []
        logits_list = []

        input_x = Batch_Normalization(input_x, training=training, scope=layer_name + '_batch_norm')
        input_x = tf.nn.relu(input_x, name=layer_name + 'relu')

        for i in range(2):  # up and down
            local_x2 = avg_pool_2d(input_x[:, i * stripe_h2: (i + 1) * stripe_h2, :, :], (stripe_h2, stripe_w2))
            local_x2 = tf.layers.dropout(inputs=local_x2, rate=drop_rate, training=training,
                                         name=layer_name + '_drop1')
            local_x2 = conv_layer(local_x2, filter=256, kernel=[1, 1],
                                  layer_name=layer_name + '_split_conv' + str(i))
            local_x2 = Batch_Normalization(local_x2, training=training, scope=layer_name + '_batch' + str(i))
            local_x2 = flatten(local_x2)
            local_x_list.append(local_x2)
            local_x2 = tf.layers.dropout(inputs=local_x2, rate=drop_rate, training=training,
                                         name=layer_name + '_drop2')
            local_x2 = Fully_connected(local_x2, class_num, use_bias=False,
                                       layer_name=layer_name + '_fully_connected_' + str(i))
            logits_list.append(local_x2)

        return local_x_list, logits_list

# straight 3 parts
def Straight_extract(input_x, layer_name, class_num, training, drop_rate=0.2):
    with tf.name_scope(layer_name):
        output_height = int(np.shape(input_x)[1])
        output_weight = int(np.shape(input_x)[2])
        stripe_w3 = int(output_weight / 3)
        stripe_h3 = output_height
        local_x_list = []
        logits_list = []
        # input_x = Batch_Normalization(input_x, training=training, scope=layer_name + '_batch_norm')

        # ***************Left Part*****************
        local_x3_1 = avg_pool_2d(input_x[:, :, :stripe_w3, :], (stripe_h3, stripe_w3))
        local_x3_1 = tf.layers.dropout(inputs=local_x3_1, rate=drop_rate, training=training,
                                       name=layer_name + '_drop11')
        local_x3_1 = conv_layer(local_x3_1, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv0')
        local_x3_1 = Batch_Normalization(local_x3_1, training=training, scope=layer_name + '_batch0')
        local_x3_1 = flatten(local_x3_1)
        local_x_list.append(local_x3_1)
        local_x3_1 = tf.layers.dropout(inputs=local_x3_1, rate=drop_rate, training=training,
                                       name=layer_name + '_drop12')
        local_x3_1 = Fully_connected(local_x3_1, class_num, use_bias=False, layer_name=layer_name + '_fully_connected_0')
        logits_list.append(local_x3_1)

        # ***************Middle Part*****************
        if output_weight % 3 != 0:
            stripe_wm = int(stripe_w3 * 2 + output_weight % 3)
        else:
            stripe_wm = stripe_w3 * 2
        local_x3_2 = avg_pool_2d(input_x[:, :, stripe_w3:stripe_wm, :], (stripe_h3, stripe_wm))
        local_x3_2 = tf.layers.dropout(inputs=local_x3_2, rate=drop_rate, training=training,
                                       name=layer_name + '_drop21')
        local_x3_2 = conv_layer(local_x3_2, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv1')
        local_x3_2 = Batch_Normalization(local_x3_2, training=training, scope=layer_name + '_batch1')
        local_x3_2 = flatten(local_x3_2)
        local_x_list.append(local_x3_2)
        local_x3_2 = tf.layers.dropout(inputs=local_x3_2, rate=drop_rate, training=training,
                                       name=layer_name + '_drop22')
        local_x3_2 = Fully_connected(local_x3_2, class_num, use_bias=False, layer_name=layer_name + '_fully_connected_1')
        logits_list.append(local_x3_2)

        # ***************Right Part*****************
        local_x3_3 = avg_pool_2d(input_x[:, :, stripe_wm:, :], (stripe_h3, stripe_w3))
        local_x3_3 = tf.layers.dropout(inputs=local_x3_3, rate=drop_rate, training=training,
                                       name=layer_name + '_drop31')
        local_x3_3 = conv_layer(local_x3_3, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv2')
        local_x3_3 = Batch_Normalization(local_x3_3, training=training, scope=layer_name + '_batch2')
        local_x3_3 = flatten(local_x3_3)
        local_x_list.append(local_x3_3)
        local_x3_3 = tf.layers.dropout(inputs=local_x3_3, rate=drop_rate, training=training,
                                       name=layer_name + '_drop32')
        local_x3_3 = Fully_connected(local_x3_3, class_num, use_bias=False, layer_name=layer_name + '_fully_connected_2')
        logits_list.append(local_x3_3)

        return local_x_list, logits_list

def Partbased(net, num_classes, layer_name='', is_training=False, num_dims=256, drop_rate=0.2,
              part=False, all=False, half=False, straight=False):
    features_list = []
    pred_list = []

    if part:
        local_x_list_6, logits_6 = Part_extract(net, layer_name=layer_name+'part',
                                                class_num=num_classes, training=is_training, drop_rate=drop_rate)

        features_list.extend(local_x_list_6)
        pred_list.extend(logits_6)

    if all:
        local_x_list_1, logits_1 = All_extract(net, layer_name=layer_name+'all', num_dims=num_dims,
                                               class_num=num_classes, training=is_training, drop_rate=drop_rate)
        features_list.extend(local_x_list_1)
        pred_list.extend(logits_1)

    if half:
        local_x_list_2, logits_2 = Half_extract(net, layer_name=layer_name+'half',
                                                class_num=num_classes, training=is_training, drop_rate=drop_rate)
        features_list.extend(local_x_list_2)
        pred_list.extend(logits_2)

    if straight:
        local_x_list_3, logits_3 = Straight_extract(net, layer_name=layer_name+'straight',
                                                    class_num=num_classes, training=is_training, drop_rate=drop_rate)
        features_list.extend(local_x_list_3)
        pred_list.extend(logits_3)

    return features_list, pred_list