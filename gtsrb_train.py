#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import random
import cv2
import time
import cPickle as pickle

import tensorflow as tf
from tensorflow.contrib.layers import flatten

# 加载mnist_inference.py中定义的常量和前向传播的函数
import gtsrb_inference

# 配置神经网络的参数
BATCH_SIZE = 2
EPOCHS = 30000


def LeNet(x):
    mu = 0
    sugma = 0.1

    # LAYER 1:
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1   = tf.nn.relu(conv1)

    conv1   = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # LAYER 2:
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(x, conv1, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2   = tf.nn.relu(conv2)

    conv2   = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0     = flatten(conv2)

    fc1_w   = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b   = tf.Variable(tf.zeros(120))
    fc1     = tf.matmul(fc0, fc1_w) + fc1_b

    fc1     = tf.nn.relu(fc1)

    fc2_w   = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b   = tf.Variable(tf.zeros(120))
    fc2     = tf.matmul(fc1, fc2_w) + fc2_b

    fc2     = tf.nn.relu(fc2)

    fc3_w   = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b   = tf.Variable(tf.zeros(120))
    fc3     = tf.matmul(fc2, fc3_w) + fc3_b

    logits  = fc3
    return logits

def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./dataset/MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    return X_train, y_train, X_validation, y_validation, X_test, y_test

X_train, y_train, X_validation, y_validation, X_test, y_test = load_data()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
train_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset + BATCH_SIZE], y_data[offset : offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset : end], y_train[offset : end]
            sess.run(train_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {0}/{1} ...".format(i+1, EPOCHS))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save("sess", 'lenet')
    print("Model saved")

exit(0)

LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"

def train(gtsrb_items):
    # 定义输入输出placeholder
    # 调整输入数据placeholder的格式，输入为一个四维矩阵
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,                             # 第一维表示一个batch中样例的个数
        gtsrb_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
        gtsrb_inference.IMAGE_SIZE,
        gtsrb_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, gtsrb_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = gtsrb_inference.inference(x, True, regularizer)
    
    global_step = tf.Variable(0, trainable=False)

    #定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    num_examples = 30000
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 验证和测试的过程将会有一个独立的程序来完成
        for i in range(EPOCHS):
            xs = np.zeros((BATCH_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.NUM_CHANNELS))
            ys = np.zeros((BATCH_SIZE, gtsrb_inference.NUM_LABELS))
            for j in range(BATCH_SIZE):
                item = random.choice(gtsrb_items)
                image = cv2.imread(item['image_fn'], 1)
                x1, y1, x2, y2 = item['bbox']
                image = image[x1:x2,y1:y2]
                image = cv2.resize(image, (gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE))
                image = (image - 128.0) / 256.0
                xs[j,:,:, :] = image[:, :, :]
                class_id = np.zeros(gtsrb_inference.NUM_LABELS)
                class_id[item['class_id']] = 1.0
                if not i and j < 10:
                    print class_id, item['class_id'], item['image_fn']
                ys[j, :] = class_id
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
            # reshaped_xs = np.reshape(xs, (BATCH_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.NUM_CHANNELS))
            # print xs.shape, ys.shape
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            #每1000轮保存一次模型。
            if (i%1000 == 0) or (i == EPOCHS-1):
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print("[%s]After %d training step(s), loss on training batch is %f." % (time_str, step, loss_value))
                # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    # train length : 34799
    # test length  : 12630
    # valid length :  4410
    train_fn = './dataset/valid.2.p'
    with open(train_fn, 'rb') as fd:
        train_data = pickle.load(fd)
        print train_data.keys()
        print len(train_data['labels']), len(train_data['features']), len(train_data['coords']), len(train_data['sizes'])

if __name__ == '__main__':
    tf.app.run()
