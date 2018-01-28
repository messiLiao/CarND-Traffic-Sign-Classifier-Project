#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import random
import cv2
import time

import tensorflow as tf

# 加载mnist_inference.py中定义的常量和前向传播的函数
import gtsrb_inference

# 配置神经网络的参数
BATCH_SIZE = 2
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
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
        for i in range(TRAINING_STEPS):
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
            if i%1000 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                print("[%s]After %d training step(s), loss on training batch is %f." % (time_str, step, loss_value))
                # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def generate_json_file(train_json_fn, test_json_fn):
    dataset_path = os.path.join(".", "dataset", "GTSRB", "Final_Training", "Images")
    subdirs = os.listdir(dataset_path)
    gtsrb_train_items = []
    gtsrb_test_items = []
    item_count = 0
    for subdir in subdirs:
        fulldir = os.path.join(dataset_path, subdir)
        subfns = os.listdir(fulldir)
        image_fns = [os.path.join(fulldir, fn) for fn in subfns if fn.endswith('.ppm')]
        item_count += len(image_fns)
        annotations_fns = [os.path.join(fulldir, fn) for fn in subfns if fn.endswith('.csv')]
        assert(len(annotations_fns) == 1)
        assert(len(image_fns)+len(annotations_fns) == len(subfns))
        annotations_fn = annotations_fns[0]
        gtsrb_items = []
        with open(annotations_fn, 'r') as fd:
            lines = fd.readlines()
            for i, line in enumerate(lines):
                params = line.replace("\r", '').replace('\n', '').split(';')
                if i > 0:
                    fn, w, h, x1, y1, x2, y2, class_id = params
                    fn, w, h, x1, y1, x2, y2, class_id = fn, int(w), int(h), int(x1), int(y1), int(x2), int(y2), int(class_id)
                    item = {}
                    item['image_fn'] = os.path.join(fulldir, fn)
                    item['size'] = (w, h)
                    item['bbox'] = (x1, y1, x2, y2)
                    item['class_id'] = class_id
                    gtsrb_items.append(item)
        test_cnt = int(len(gtsrb_items) * 0.1)
        test_items = [gtsrb_items.pop(random.randint(0, len(gtsrb_items) - 1)) for i in range(test_cnt)]

        gtsrb_test_items += test_items
        gtsrb_train_items += gtsrb_items
    with open(train_json_fn, 'w') as fd:
        json.dump(gtsrb_train_items, fd)
    with open(test_json_fn, 'w') as fd:
        json.dump(gtsrb_test_items, fd)

def main(argv=None):
    train_json_fn = './gtsrb_train.json'
    test_json_fn = './gtsrb_test.json'
    if not os.path.isfile(train_json_fn):
        generate_json_file(train_json_fn, test_json_fn)
    
    with open(train_json_fn, 'r') as fd:
        gtsrb_train_items = json.load(fd)
    train(gtsrb_train_items)


if __name__ == '__main__':
    tf.app.run()