import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model/lenet.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input/x:0")
y = graph.get_tensor_by_name("input/y:0")
n_classes = 43
one_hot_y = tf.one_hot(y, n_classes)

logits = graph.get_tensor_by_name('layer-fc3/fc3:0')

softmax = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name='correct_prediction')
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')

def preprocess_image(image):
    shape = image.shape
    if len(shape) == 2:
        shape = (32, 32, 1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = (32, 32, 1)
    image = cv2.resize(image, (32, 32))
    image = cv2.equalizeHist(image)
    min_val = np.min(image) * 1.0
    max_val = np.max(image) * 1.0
    image = (image - min_val) / max_val - 0.5
    return image.reshape(shape)

def plot_and_save(data_list, title, save_path='', show=False):
    plt.figure()
    x = [d[1] for d in data_list]
    y = []
    min_show_value = 0.01
    for d in data_list:
        y.append(d[0] if d[0] > min_show_value else min_show_value)
    plt.bar(x, y)
    plt.xlabel('class')
    plt.ylabel('Probability')
    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    if(show):
        plt.show()

X_train = np.zeros((6, 32, 32, 1))
y_train = np.zeros((6))
index = 0
for i in range(7):
    fn = './examples/traffic_sign_%d.png' % i
    if i == 4:
        continue
    image = cv2.imread(fn)
    X_train[index] = preprocess_image(image).reshape((1, 32, 32, 1))
    y_train[index] = i
    index += 1

print "Accuracy:", sess.run(accuracy_operation, feed_dict={x:X_train, y:y_train})


for i in range(7):
    fn = './examples/traffic_sign_%d.png' % i
    image = cv2.imread(fn)
    cv2.imwrite(fn, image)
    X_train = preprocess_image(image).reshape((1, 32, 32, 1))
    y_train = np.zeros((1, 1))
    y_train[0, 0] = 0
    result = sess.run(softmax, feed_dict={x:X_train, y:y_train})
    result_index = [(result[0, j], j) for j in range(result.shape[1])]
    top_5 = sorted(result_index, key=lambda x:x[0], reverse=True)[:5]
    plot_and_save(top_5, 'Top 5 Probability of predict in %dth image' % i, './examples/probability_top5_of_%d.png' % i, False)
    predict = np.argmax(result)

    print "fn=%s, class=%d, Probability=%.4f" % (fn, predict, result[0, predict])

