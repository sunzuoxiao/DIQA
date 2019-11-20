# coding=utf-8
# created by 'szx' on '12/11/19'
from live_utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import os
from PIL import Image
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def read_image_label_tfrecord(filename, image_width, image_heigh):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_file = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_file,
                                       features={

                                           'I_d_data': tf.FixedLenFeature([], tf.string),
                                           'e_gt_data': tf.FixedLenFeature([], tf.string),
                                           'r_data': tf.FixedLenFeature([], tf.string),
                                           'ori_data':tf.FixedLenFeature([], tf.string),
                                           'label_data': tf.FixedLenFeature([], tf.int64),
                                           'psnr_data': tf.FixedLenFeature([], tf.int64)
                                       })

    I_d_data = tf.decode_raw(features['I_d_data'], tf.float32)
    I_d_data = tf.reshape(I_d_data, [image_width, image_heigh, 1])

    e_gt_data = tf.decode_raw(features['e_gt_data'], tf.float32)
    e_gt_data = tf.reshape(e_gt_data, [int(image_width/4), int(image_heigh/4), 1])

    r_data = tf.decode_raw(features['r_data'], tf.float32)
    r_data = tf.reshape(r_data, [int(image_width/4), int(image_heigh/4), 1])

    ori_data = tf.decode_raw(features['ori_data'], tf.uint8)
    ori_data = tf.reshape(ori_data, [image_width, image_heigh, 3])
    ori_data = tf.image.convert_image_dtype(ori_data, dtype=tf.float32)


    return I_d_data, e_gt_data, r_data, ori_data


def image_label_batch_queue_tfrecord(filename,
                                     image_width,
                                     image_heigh,
                                     shuffle=True,
                                     batch_size=32,
                                     num_threads=1,
                                     min_after_dequeue=10):

    I_d_data, e_gt_data,r_data, ori_data = read_image_label_tfrecord(filename, image_width, image_heigh)

    if shuffle:

        I_d_data_batch, e_gt_data_batch,r_data_batch, ori_data_batch = \
            tf.train.shuffle_batch([I_d_data, e_gt_data,r_data, ori_data],
                                   batch_size=batch_size,
                                   capacity=min_after_dequeue + 3 * batch_size,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=num_threads)

    else:
        I_d_data_batch, e_gt_data_batch, r_data_batch, ori_data_batch = \
            tf.train.batch([I_d_data, e_gt_data,r_data, ori_data],
                                            batch_size=batch_size,
                                            capacity=min_after_dequeue + 3 * batch_size,
                                            num_threads=num_threads)

    return  I_d_data_batch, e_gt_data_batch,r_data_batch, ori_data_batch


def calculate_error_map(I_d_data, I_r_data):

    I_d = image_preprocess(I_d_data)

    I_r = image_preprocess(I_r_data)

    r = rescale(average_reliability_map(I_d, 0.2), 1 / 4)

    e_gt = rescale(error_map(I_r, I_d, 0.2), 1 / 4)

    return I_d, e_gt, r


def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


def tfrecord_image_label_make(output_path=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    writer = tf.python_io.TFRecordWriter(output_path)
    root_image_path = '/media/szx/新加卷2/DIQA_224/'
    list_level = ['level_00', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5']
    i = 0
    # 失真图像
    I_d_data = tf.placeholder(tf.float32, shape=[None, None, 3])
    # 参考图像
    I_r_data = tf.placeholder(tf.float32, shape=[None, None, 3])

    I_d, e_gt, r = calculate_error_map(I_d_data, I_r_data)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for index_i in range(1, 9500):

            I_r_path = root_image_path + 'level_00/' + 'index_' + str(index_i) + '.png'
            I_r_read = plt.imread(I_r_path)
            psnr_I_r_data = cv2.imread(I_r_path)
            for level_mun in range(1, 6):

                I_d_path = root_image_path + list_level[level_mun] + '/' + 'index_' + str(index_i) + '.png'

                I_d_read = plt.imread(I_d_path)

                psnr_I_d_data = cv2.imread(I_d_path)
                I_d_read_img = Image.open(I_d_path)

                feed_dict = {
                    I_d_data: I_d_read,
                    I_r_data: I_r_read
                }

                I_d_data_result, e_gt_data_result, r_data_result = sess.run([I_d, e_gt, r], feed_dict=feed_dict)

                I_d_data_byte = I_d_data_result[0].tobytes()
                e_gt_data_byte = e_gt_data_result[0].tobytes()
                r_data_byte = r_data_result[0].tobytes()
                I_d_read_byte = I_d_read_img.tobytes()

                file_record = tf.train.Example(features=tf.train.Features(feature={

                    'I_d_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[I_d_data_byte])),
                    'e_gt_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[e_gt_data_byte])),
                    'r_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[r_data_byte])),
                    'ori_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[I_d_read_byte])),

                }))

                writer.write(file_record.SerializeToString())

        i = i + 1
    print('finished, total image is %s ', i)

    writer.close()


# For weight decay,L 2 regularization was applied to all the layers (L 2 penalty multiplied by 5 × 10 −4 ).
def conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=True, strides=1):

    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
        use_bias=use_bias, name=name)
    return conv

def bottleneck(inputs, depth, expand=1, stride=1, use_BN=True, scope="bottleneck"):

  with tf.variable_scope(scope):

    shortcut = slim.conv2d(inputs, depth*expand, [1, 1], stride=stride, scope='shortcut')

    residual = slim.conv2d(inputs, depth, [1, 1], stride=1, scope='conv1')
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(residual, depth, [3, 3], stride=stride,scope='conv2')
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(residual, depth*expand, [1, 1], stride=1, scope='conv3')
    output = shortcut + residual
    if use_BN:
        output = slim.batch_norm(output, decay=0.99, activation_fn=tf.nn.relu, scope='norm')
    else:
        output = tf.nn.relu(residual)
    return output


def DIQA_model(input,is_train=True):

    with tf.variable_scope('train_main'):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001),
                            activation_fn=None,
                            biases_initializer=tf.constant_initializer(0.1)):
            with slim.arg_scope([slim.batch_norm], is_training=is_train):
                conv1 = slim.conv2d(input, 64, 7, stride=1, scope='conv1')
                conv1 = tf.nn.relu(conv1)

                bottleneck1 = bottleneck(conv1, 64, expand=4, stride=2, use_BN=False, scope="bottleneck1")
                bottleneck2 = bottleneck(bottleneck1, 128, expand=4, stride=1, use_BN=False, scope="bottleneck2")

                # error map caculate 128*4 = 512

                bottleneck3 = bottleneck(bottleneck2, 256, expand=4, stride=2, scope="bottleneck3")

                error_map = slim.conv2d(bottleneck3, 1, [1, 1], stride=1, scope='error_map')

                net = bottleneck(bottleneck3, 256, expand=4, stride=2, scope="bottleneck4")
                #
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                net = slim.conv2d(net, 5, [1, 1], scope='logits')
                # # net = tf.nn.dropout(net, dropout_keep_prob)
                net_class = tf.squeeze(net, [1, 2], name='SpatialSqueeze')


    return net_class, error_map

def DIQA_h_f_optimize(input, num_classes, weight, scope="resnet_v2_cpu", is_train=False):

    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001),
                            activation_fn=None,
                            biases_initializer=tf.constant_initializer(0.1)):

            net = slim.conv2d(input, 16, 7, stride=1, scope='conv1')
            net = tf.nn.relu(net)

            # net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            # net = slim.batch_norm(net, decay=config.norm_decay, activation_fn=tf.nn.relu, scope='norm1')

            net = bottleneck(net, 16, expand=4, stride=1, use_BN=False, scope="bottleneck1")
            net = bottleneck(net, 32, expand=4, stride=1, use_BN=False, scope="bottleneck2")
            net = bottleneck(net, 32, expand=4, stride=1, use_BN=False, scope = "bottleneck3")
            error_map = slim.conv2d(net, 1, [3, 3], stride=1, scope='error_map')
            print(error_map)
            class_map = error_map * weight

            net_class = slim.max_pool2d(class_map, [3, 3], stride=4, scope='pool5')

            net_class = tf.layers.Flatten()(net_class)

            net_logit = tf.layers.dense(net_class, num_classes)


    return input, error_map, net_logit


def sigmoid_loss(label, logit):

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)

    return loss_

def loss_error_map(y_pred, y_true, r):

    return tf.reduce_mean(tf.square((y_true - y_pred) * r))*10.0


def loss_score(pred_score, true_score):

    return tf.reduce_mean(tf.square(true_score - pred_score))


def train_error_map():

    tf_record_path = ''
    model_dir = ''

    I_d_data_batch, e_gt_data_batch, r_data_batch, ori_data_batch = \
        image_label_batch_queue_tfrecord(
        filename=tf_record_path,
        image_width=224,
        image_heigh=224,
        shuffle=True)

    image_d = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    true_map = tf.placeholder(tf.float32, shape=[None, 56, 56, 1])
    weight_r = tf.placeholder(tf.float32, shape=[None, 56, 56, 1])
    is_training = tf.placeholder(tf.bool, shape=[])
    input_label_hl = tf.placeholder(tf.int32, shape=[None])

    net_class, error_map = DIQA_model(input=image_d,is_train=is_training)
    pre = tf.nn.softmax(net_class)

    loss_map = loss_error_map(y_pred=error_map, y_true=true_map, r=weight_r)
    loss_class = sigmoid_loss(logit=net_class,label=input_label_hl)
    loss_vaule = loss_map + loss_class

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.0005, global_step, decay_steps=5000, decay_rate=0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss_vaule, global_step=global_step, var_list=tf.global_variables())

    init = tf.global_variables_initializer()
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    start_time = time.time()
    tf_config = tf.ConfigProto()

    with tf.Session(config=tf_config) as sess:

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('restore model', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print(' no pretrained, model path is ', model_dir)
            sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(5):
            print('current epoch', epoch)
            print('used time :', (time.time() - start_time))
            start_time = time.time()

            for step in range(int(5)):

                I_d_data_b, e_gt_data_batch_b, r_data_batch_b,  = sess.run([I_d_data_batch,
                                                                            e_gt_data_batch,
                                                                            r_data_batch])

                feed_dict = {
                    image_d: I_d_data_b,
                    true_map:e_gt_data_batch_b,
                    weight_r:r_data_batch_b,

                    is_training:True
                }

                train_loss,train_loss_map, train_loss_class, _, _lr, logit_result=sess.run([loss_vaule,loss_map, loss_class, train_op, lr,pre],
                                                                               feed_dict=feed_dict)
                if step % 100 == 0:
                    print('loss:', train_loss, 'loss_map',train_loss_map,'train_class',train_loss_class,'current lr；', _lr)
                    output_label = np.argmax(logit_result, axis=1)
                    print(output_label)



            # 每5个epoch保存一次模型
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')

                saver.save(sess, checkpoint_path, global_step=global_step)

            # freeze mode to pb
            if epoch == (50 - 1):
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                print('save checkpoint,the global_step is ', global_step)
                print('save path:', checkpoint_path)
                saver.save(sess, checkpoint_path, global_step=global_step)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":


   train_error_map()






