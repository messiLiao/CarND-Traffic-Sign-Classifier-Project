#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
import json
import cv2

# 加载gtsrb_inference.py 和 gtsrb_train.py中定义的常量和函数
import gtsrb_inference
import gtsrb_train

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 100


def evaluate(gtsrb_items):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        num_examples = len(gtsrb_items)
        x = tf.placeholder(tf.float32, [
            1,           # 第一维表示样例的个数
            gtsrb_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
            gtsrb_inference.IMAGE_SIZE,
            gtsrb_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
        y_ = tf.placeholder(tf.float32, [None, gtsrb_inference.OUTPUT_NODE], name='y-input')
        test_images = np.zeros((num_examples, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.NUM_CHANNELS))
        test_labels = np.zeros((num_examples, gtsrb_inference.NUM_LABELS))
        for i, item in enumerate(gtsrb_items):
            image_fn = item['image_fn']
            x1, y1, x2, y2 = item['bbox']
            class_id = item['class_id']
            image = cv2.imread(image_fn, 1)
            image = image[x1:x2, y1:y2]
            image = cv2.resize(image, (gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE))
            test_images[i, :, :] = image
            label = np.zeros(gtsrb_inference.NUM_LABELS)
            label[item['class_id']] = 1.0
            test_labels[i] = label
        validate_feed = {x: np.reshape(test_images, (num_examples, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.NUM_CHANNELS)),
                         y_: test_labels}
        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。
        y = gtsrb_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用gtsrb_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(gtsrb_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        config = tf.ConfigProto(device_count = {'GPU': 0} )
        while True:
            with tf.Session(config=config) as sess:
                    # with tf.device("/cpu:0"):
                    # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                    ckpt = tf.train.get_checkpoint_state(gtsrb_train.MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        # 加载模型
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        # 通过文件名得到模型保存时迭代的轮数
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        gtsrb_total_count = len(gtsrb_items)
                        gtsrb_right_count = 0
                        for i, item in enumerate(gtsrb_items):
                            image_fn = item['image_fn']
                            x1, y1, x2, y2 = item['bbox']
                            class_id = item['class_id']
                            image = cv2.imread(image_fn, 1)
                            image = image[x1:x2, y1:y2]
                            image = cv2.resize(image, (gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE))
                            test_images[i, :, :] = image
                            label = np.zeros((1, gtsrb_inference.NUM_LABELS))
                            label[0, item['class_id']] = 1.0
                            test_labels[i] = label
                            validate_feed = {x: np.reshape(image, (1, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.IMAGE_SIZE, gtsrb_inference.NUM_CHANNELS)),
                             y_: label}
                            is_correct = sess.run(correct_prediction, feed_dict = validate_feed)
                            if is_correct[0]:
                                gtsrb_right_count += 1
                        print("After %s training step(s), validation accuracy = %f" % (global_step, gtsrb_right_count * 1.0 / gtsrb_total_count))
                    else:
                        print("No checkpoint file found")
                        return
            break
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):

    train_json_fn = './gtsrb_train.json'
    test_json_fn = './gtsrb_test.json'
    
    with open(test_json_fn, 'r') as fd:
        gtsrb_test_items = json.load(fd)
    evaluate(gtsrb_test_items)


if __name__ == '__main__':
    tf.app.run()