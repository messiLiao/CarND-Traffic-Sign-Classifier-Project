import cv2
import numpy as np
import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model/lenet.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input/x:0")
y = graph.get_tensor_by_name("input/y:0")

accuracy_operation = graph.get_tensor_by_name('layer-fc3/fc3:0')

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

# sess.run('fc3:0', feed_dict={'input-x:0':0, 'input-y:0':0})
image = cv2.imread('./examples/traffic_sign_0.png')
X_train = preprocess_image(image).reshape((1, 32, 32, 1))
y_train = np.zeros((1, 1))
y_train[0, 0] = 0
result = sess.run(accuracy_operation, feed_dict={x:X_train, y:y_train})
print np.argmax()