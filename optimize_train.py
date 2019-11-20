# coding=utf-8
# created by 'szx' on '20/11/19'
from live_diqa_main import *
import numpy as np
import time


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


def loss_h_f_imformation(input, y_true, r, error_map, net_logit, label):
    print('ttttttttttttttttttttttttt')
    print(input)
    print(error_map)

    # 考虑权重r  是否放在error_map之后  相乘后在进行池化， 进而全连接分层。

    error_map_value = tf.reduce_mean(tf.square(y_true - (input + error_map)))*1000
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_logit, labels=label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)

    return error_map_value, loss_


Epoch = 50
filename_tfrecord = 'train.tfrecord'
num_classes = 5
learning_rate = 0.0001
model_dir = 'model'
epoch_num = 1000


def train_error_map():


    I_d_data_batch, I_r_data_batch, e_gt_data_batch, r_data_batch, ori_data_batch, label_data_batch, psnr_data_batch = \
        image_label_batch_queue_tfrecord(
        filename=filename_tfrecord,
        image_width=224,
        image_heigh=224,
        shuffle=True)

    image_d = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    true_map = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    weight_r = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    is_training = tf.placeholder(tf.bool, shape=[])
    input_label_hl = tf.placeholder(tf.int32, shape=[None])
    loss_weight = tf.placeholder(tf.float32)

    input, error_map, net_class = DIQA_h_f_optimize(input=image_d,
                                                    num_classes=num_classes,
                                                    weight=weight_r,
                                                    is_train=is_training)
    pre = tf.nn.softmax(net_class)

    print('ddddddddddddddddddddddddddddddddddddddddddd')
    print(pre)

    loss_map, loss_class = loss_h_f_imformation(input=input,
                                                y_true=true_map,
                                                r=weight_r,
                                                error_map=error_map,
                                                net_logit=net_class,
                                                label=input_label_hl)

    loss_vaule = (Epoch/5 - loss_weight)*loss_map + loss_weight*loss_class

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps=5000, decay_rate=0.95)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss_vaule, global_step=global_step, var_list=tf.global_variables())

    init = tf.global_variables_initializer()
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    start_time = time.time()
    tf_config = tf.ConfigProto()

    weight_value = 1

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

        for epoch in range(Epoch):
            print('current epoch', epoch)
            print('used time :', (time.time() - start_time))
            start_time = time.time()


            for step in range(int(epoch_num)):

                I_d_data_b, I_r_data_batch_b, r_data_batch_b, label_data_batch_b = sess.run([I_d_data_batch,
                                                                                              I_r_data_batch,
                                                                                              r_data_batch,
                                                                                              label_data_batch])

                feed_dict = {
                    image_d: I_d_data_b,
                    true_map:I_r_data_batch_b,
                    weight_r:r_data_batch_b,
                    input_label_hl:label_data_batch_b,
                    loss_weight:weight_value,
                    is_training:True
                }

                train_loss,train_loss_map, train_loss_class, _, _lr, logit_result,loss_weight_ = \
                    sess.run([loss_vaule,loss_map, loss_class, train_op, lr, pre, loss_weight],
                                                                               feed_dict=feed_dict)
                if step % 100 == 0:

                    print('loss:', train_loss, 'loss_map',train_loss_map,'train_class',train_loss_class,'loss_weight_',loss_weight_,'current lr；', _lr)
                    print('error map',(Epoch / 5 - loss_weight_)*train_loss_map,'train_class_weight',loss_weight_*train_loss_class)
                    output_label = np.argmax(logit_result, axis=1)
                    print(output_label)
                    print(label_data_batch_b)


            # 每5个epoch保存一次模型
            if epoch % 5 == 0:
                weight_value = weight_value + 1
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            # freeze mode to pb
            if epoch == (Epoch - 1):
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                print('save checkpoint,the global_step is ', global_step)
                print('save path:', checkpoint_path)
                saver.save(sess, checkpoint_path, global_step=global_step)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":

   train_error_map()






