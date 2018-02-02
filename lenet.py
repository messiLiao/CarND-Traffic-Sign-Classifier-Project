#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import random
import cv2
import time
import cPickle as pickle
import sklearn

import tensorflow as tf
from tensorflow.contrib.layers import flatten

BATCH_SIZE = 128
EPOCHS = 50
learning_rate = 0.001

data_source = 'gtsrb'  # 'gtsrb' or mnist

if data_source == 'gtsrb':
    n_classes = 43
elif data_source == 'mnist':
    n_classes = 10
else:
    print("Unsupport data source. gtsrb or mnist is supported.")
    exit(-1)


def LeNet(x):
    mu = 0
    sigma = 0.1

    # LAYER 1:
    with tf.variable_scope('layer-conv1'):
        conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma), name='conv1_w')
        conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
        conv1   = tf.add(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID'), conv1_b, name='conv1')

        conv1   = tf.nn.relu(conv1, name='conv1-relu')

        conv1   = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv1-maxpool')
        print conv1.shape


    # LAYER 2:
    with tf.variable_scope('layer-conv2'):
        conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name='conv2_w')
        conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
        conv2   = tf.add(tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID'), conv2_b, name='conv2')

        conv2   = tf.nn.relu(conv2, name='conv2-relu')

        conv2   = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv2-maxpool')
        print conv2.shape

    # LAYER 3
    fc0     = flatten(conv2)

    # LAYER 4
    with tf.variable_scope('layer-fc1'):
        fc1_w   = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name='w')
        fc1_b   = tf.Variable(tf.zeros(120), name='b')
        fc1     = tf.add(tf.matmul(fc0, fc1_w), fc1_b, name='fc1')

        fc1     = tf.nn.relu(fc1, name='relu')

    # LAYER 5
    with tf.variable_scope('layer-fc2'):
        fc2_w   = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name='w')
        fc2_b   = tf.Variable(tf.zeros(84), name='b')
        fc2     = tf.add(tf.matmul(fc1, fc2_w), fc2_b, name='fc2')

        fc2     = tf.nn.relu(fc2, name='relu')

    # LAYER 6
    with tf.variable_scope('layer-fc3'):
        fc3_w   = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma), name='w')
        fc3_b   = tf.Variable(tf.zeros(n_classes), name='b')
        fc3     = tf.add(tf.matmul(fc2, fc3_w), fc3_b, name='fc3')

    logits  = fc3
    return logits

def load_data(source='mnist'):
    if data_source == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("./dataset/MNIST_data/", reshape=False)
        X_train, y_train = mnist.train.images, mnist.train.labels
        X_validation, y_validation = mnist.validation.images, mnist.validation.labels
        X_test, y_test = mnist.test.images, mnist.test.labels
    elif data_source == 'gtsrb':
        train_fn = './dataset/traffic-signs-data/train.2.p'
        with open(train_fn, 'rb') as fd:
            train_data = pickle.load(fd)
            X_train, y_train = train_data['features'], train_data['labels']
        test_fn = './dataset/traffic-signs-data/test.2.p'
        with open(test_fn, 'rb') as fd:
            test_data = pickle.load(fd)
            X_test, y_test = test_data['features'], test_data['labels']
        valid_fn = './dataset/traffic-signs-data/valid.2.p'
        with open(valid_fn, 'rb') as fd:
            valid_data = pickle.load(fd)
            X_validation, y_validation = valid_data['features'], valid_data['labels']
            print X_validation.shape, y_validation.shape
    else:
        return None, None, None, None, None, None
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def preprocess_image(image):
    shape = image.shape
    if len(shape) == 2:
        shape = (32, 32, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = (32, 32, 1)
    image = cv2.equalizeHist(image)
    min_val = np.min(image) * 1.0
    max_val = np.max(image) * 1.0
    image = (image - min_val) / max_val - 0.5
    return image.reshape(shape)

X_train, y_train, X_validation, y_validation, X_test, y_test = load_data(data_source)
try:
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.hist(y_train)
    plt.xlabel("class")
    plt.ylabel("count")
    plt.title("number of each calss samples in train dataset")
    plt.savefig("./examples/hist_train.png")

    plt.figure(2)
    plt.hist(y_validation)
    plt.title("number of each calss samples in validation dataset")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.savefig("./examples/hist_validation.png")
    plt.figure(3)
    plt.hist(y_test)
    plt.xlabel("class")
    plt.ylabel("count")
    plt.title("number of each calss samples in test dataset")
    plt.savefig("./examples/hist_test.png")
    # plt.show()
except Exception, err:
    print(err.message)
#     pass
# for i in range(10):
#     index = random.randint(0, len(X_train))
#     image = X_train[index]
#     cv2.imshow('image', image)
#     cv2.imshow('preprocess', preprocess_image(image))
#     cv2.waitKey(0)
# exit(0)
# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_validation)

# Number of testing examples.
n_test = len(X_test)

# the shape of an traffic sign image:
image_shape = (32, 32, 3)

print("Number of train samples:%d" % n_train)
print("Number of test samples:%d" % n_test)
print("number of validation samples:%d" % n_validation)

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='x')
    y = tf.placeholder(tf.int32, (None), name='y')
    one_hot_y = tf.one_hot(y, n_classes)


logits = LeNet(x)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(one_hot_y, 1), name='cross_entropy')
loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name='correct_prediction')
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset + BATCH_SIZE], y_data[offset : offset + BATCH_SIZE]
        n, w, h, c = batch_x.shape
        new_batch_x = np.zeros((n, 32, 32, 1))
        for j in range(n):
            new_batch_x[j] = preprocess_image(batch_x[j])
        accuracy = sess.run(accuracy_operation, feed_dict={x:new_batch_x, y:batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print("\n")

    for i in range(EPOCHS):
        print("EPOCH {0}/{1} ...".format(i+1, EPOCHS))
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset : end], y_train[offset : end]
            n, w, h, c = batch_x.shape
            new_batch_x = np.zeros((n, 32, 32, 1))
            for j in range(n):
                new_batch_x[j] = preprocess_image(batch_x[j])
            sess.run(train_operation, feed_dict={x: new_batch_x, y: batch_y})
            # print("[EPOCH=%d]%d / %d" % (i, offset, num_examples))

        validation_accuracy = evaluate(X_validation, y_validation)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    print("Test model...")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, './model/lenet')
    print("Model saved")

exit(0)
