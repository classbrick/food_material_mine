# imports
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import net.mobilenet_v2 as mobilenet_v2
import tensorflow.contrib.slim as slim
# import tensorflow.python.ops.control_flow_ops as control_flow_ops
from tf_cnnvis import *

keep_prob = tf.placeholder(dtype=tf.float32)
tf.reset_default_graph()
global_step = tf.Variable(0, trainable=True)
is_training = False
# reading data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
learning_rate = tf.train.exponential_decay(learning_rate=0.1,
                                           global_step=global_step,
                                           decay_steps=10, decay_rate=0.9)

# defining TF model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# x_image, y_conv, keep_prob = deepnn(x)
net_obj = mobilenet_v2.mobilenet_v2(is_training=is_training, scope='mobilenet_v2', num_classes=10)
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_conv = net_obj.def_net(x_image)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# trainning CNN
sess = tf.Session()

sess.run(tf.global_variables_initializer())
with sess.as_default():
    i = 0
    if is_training:
        while True:
            batch = mnist.train.next_batch(50)
            i += 1
            if i % 20 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                if train_accuracy > 0.9:
                    print('accuracy>0.9, save the model')
                    break
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    else:
        batch = mnist.train.next_batch(50)
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print("eval training accuracy %g" % train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    temp1 = batch[0][1:2]
    temp2 = batch[1][1:2]
    feed_dict = {x: batch[0][1:2], y_: batch[1][1:2]}


# deconv visualization
layers = ["r", "p", "c"]
total_time = 0

start = time.time()
is_success = deconv_visualization(sess_graph_path=sess, value_feed_dict=feed_dict,
                                  input_tensor=x_image, layers=layers,
                                  path_logdir=os.path.join("Log", "MobilenetV2"),
                                  path_outdir=os.path.join("Output", "MobilenetV2"))
start = time.time() - start
print("Total Time = %f" % (start))