import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import json
import numpy as np
import random
import cv2
import os

EPOCHS = 20
BATCH_SIZE = 64

IMAGE_SIZE = 50
NUM_CHANNELS = 1
NUM_LABELS = 43

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 50x50x1. Output = 46x46x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 46x46x6. Output = 23x23x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input = 23x23x6. Output = 19x19x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 19x19x16. Output = 10x10x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Pooling. Input = 10x10x16. Output = 5x5x32.
    conv3_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 32], mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3 = tf.nn.conv2d(conv2, conv3_w, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Pooling. Input = 5x5x32. Output = 3x3x32.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv3)
    
    # Layer 3: Fully Connected. Input = 288. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(288, 160), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(160))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 160. Output = 60.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(160, 80), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(80))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 80. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(80, NUM_LABELS), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(NUM_LABELS))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

def train():
    rate = 0.001


    train_json_fn = './gtsrb_train.json'
    test_json_fn = './gtsrb_test.json'
    with open(train_json_fn, 'r') as fd:
        gtsrb_train_items = json.load(fd)

    test_json_fn = './gtsrb_test.json'
    with open(test_json_fn, 'r') as fd:
        gtsrb_test_items = json.load(fd)
    # X_validation, y_validation = mnist.validation.images, mnist.validation.labels

    x = tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 1), name='x-input')
    y = tf.placeholder(tf.int32, (None), name='y-input')
    one_hot_y = tf.one_hot(y, NUM_LABELS)
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(gtsrb_train_items)
        test_examples = len(gtsrb_test_items)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            total_error_sample = []
            for offset in range(0, num_examples, BATCH_SIZE):
                xs = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
                ys = np.zeros((BATCH_SIZE,))
                for j in range(BATCH_SIZE):
                    item = random.choice(gtsrb_train_items)
                    image = cv2.imread(item['image_fn'], 1)
                    x1, y1, x2, y2 = item['bbox']
                    # image = image[x1:x2,y1:y2]
                    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                    if NUM_CHANNELS == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        for ii in range(200):
                            _x = int(random.uniform(0, IMAGE_SIZE))
                            _y = int(random.uniform(0, IMAGE_SIZE))
                            noisy_pix = int(random.uniform(0, 255))
                            image[_x, _y] = noisy_pix
                        image = cv2.equalizeHist(image)
                        image = (image - 128.0) / 128.0
                    else:
                        image = (image - 128.0) / 128.0
                    xs[j, :, :, 0] = image
                    ys[j] = item['class_id']


                batch_x, batch_y = xs, ys
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                # print offset, '/', num_examples

            total_accuracy = 0
            for test_offset in range(0, test_examples, BATCH_SIZE):
                batch_size = min(test_offset + BATCH_SIZE, test_examples-1) - test_offset

                test_images = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
                test_labels = np.zeros((batch_size,))
                for k in range(batch_size):
                    item = gtsrb_test_items[test_offset + k]
                    image_fn = item['image_fn']
                    x1, y1, x2, y2 = item['bbox']
                    class_id = item['class_id']
                    image = cv2.imread(image_fn, 1)
                    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                    if NUM_CHANNELS == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.equalizeHist(image)
                        test_images[k, :, :, 0] = (image - 128.0) / 128.0
                    else:
                        test_images[k, :, :] = (image - 128.0) / 128.0

                    test_labels[k] = item['class_id']

                batch_x, batch_y = test_images.astype(np.float32), test_labels
                accuracy = sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y})
                error_sample = ['%s\n'%gtsrb_test_items[test_offset+k]['image_fn'] for k, res in enumerate(accuracy) if not res]
                total_error_sample.extend(error_sample)
            print("EPOCH {} ...".format(i+1))

            print "total_accuracy=", (test_examples - len(total_error_sample)) * 1.0 / test_examples
            with open('error_samples.txt', 'w') as fd:
                fd.writelines(total_error_sample)
            print()
            
        saver.save(sess, './lenet')
        print("Model saved")

def show_error_examples():
    # 00001/00028_00005.ppm
    fn = 'error_samples.txt'
    with open(fn, 'r') as fd:
        lines = fd.readlines()
    for line in lines:
        image_fn = line.replace('\r', '').replace('\n', '')
        base, fn = os.path.split(image_fn)
        fn_list = os.listdir(base)
        label_fn = [f for f in fn_list if f.endswith('csv')][0]
        print image_fn, base, label_fn
        with open(os.path.join(base, label_fn), 'r') as fd:
            labels_lines = fd.readlines()
        for label_line in labels_lines:
            if label_line.startswith(fn):
                params = label_line.replace("\r", '').replace('\n', '').split(';')
                fn, w, h, x1, y1, x2, y2, class_id = params
                fn, w, h, x1, y1, x2, y2, class_id = fn, int(w), int(h), int(x1), int(y1), int(x2), int(y2), int(class_id)
                print fn, w, h, x1, y1, x2, y2, class_id
        origin = cv2.imread(image_fn)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray[x1:x2, y1:y2], (IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.equalizeHist(image)
        cv2.imshow('origin', origin)
        cv2.imshow('gray', gray)
        cv2.imshow('wrong', image)
        cv2.imshow('roi', roi)
        cv2.waitKey(0)

    pass

def main():
    train()
    # show_error_examples()

if __name__ == '__main__':
    main()
