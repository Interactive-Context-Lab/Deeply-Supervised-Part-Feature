import math
import numpy as np
import tensorflow as tf

def Loss_ASoftmax(x, y, l, num_cls, m=2, name='asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda
    '''

    with tf.variable_scope(name):
        xs = x.get_shape()
        ys = y.get_shape()
        w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        eps = 1e-8

        xw = tf.matmul(x,w)

        if m == 0:
            return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

        w_norm = tf.norm(w, axis=0) + eps
        logits = xw / w_norm

        if y is None:
            return logits, None

        ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
        ordinal_y = tf.stack([ordinal, y], axis=1)

        x_norm = tf.norm(x, axis = 1) + eps

        sel_logits = tf.gather_nd(logits, ordinal_y)

        cos_th = tf.div(sel_logits, x_norm)

        if m == 1:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                  (labels=y,  logits=logits))

        else:
            if m == 2:

                cos_sign = tf.sign(cos_th)
                res = 2 * tf.multiply(cos_sign, tf.square(cos_th)) - 1

            elif m == 4:

                cos_th2 = tf.square(cos_th)
                cos_th4 = tf.pow(cos_th, 4)
                sign0 = tf.sign(cos_th)
                sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
                sign4 = 2 * sign0 + sign3 - 3
                res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4

            else:
                raise ValueError('unsupported value of m')

            scaled_logits = tf.multiply(res, x_norm)

            f = 1.0 / (1.0 + l)
            ff = 1.0 - f
            comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape()))
            updated_logits = ff*logits + f*comb_logits_diff

            # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits)

    return logits, loss


def calculate_arcface_logits(embds, labels, class_num, s, m):
    weights = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], class_num],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                              regularizer=None, trainable=True)
    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t = tf.matmul(embds, weights, name='cos_t')

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
    return output


def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc': rand_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def coco_forward(xw, y, m, name=None):
    # pdb.set_trace()
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind, y] -= m
    return xw_copy


def coco_help(grad, y):
    grad_copy = grad.copy()
    return grad_copy


def coco_backward(op, grad):
    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help, [grad, y], tf.float32)
    return grad_copy, y, m


def coco_func(xw, y, m, name=None):
    with tf.op_scope([xw, y, m], name, "Coco_func") as name:
        coco_out = py_func(coco_forward, [xw, y, m], tf.float32, name=name, grad_func=coco_backward)
        return coco_out


def cos_loss(x, y, num_cls, reuse=False, alpha=0.5, scale=64, name='cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights
    xs = x.get_shape()
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

        # normalize the feature and weight
        # (N,D)
        x_feat_norm = tf.nn.l2_normalize(x, 1, 1e-10)
        # (D,C)
        w_feat_norm = tf.nn.l2_normalize(w, 0, 1e-10)

        # get the scores after normalization
        # (N,C)
        xw_norm = tf.matmul(x_feat_norm, w_feat_norm)
        # implemented by py_func
        # value = tf.identity(xw)
        # substract the marigin and scale it
        value = coco_func(xw_norm, y, alpha) * scale

    # implemented by tf api
    # margin_xw_norm = xw_norm - alpha
    # label_onehot = tf.one_hot(y,num_cls)
    # value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)

    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    return value, cos_loss