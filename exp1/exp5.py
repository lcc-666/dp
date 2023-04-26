import tensorflow as tf
import random
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

random.seed()
rowCount = 5

xData = np.full(shape=(rowCount, 3), fill_value=0, dtype=np.float32)
yTrainData = np.full(shape=rowCount, fill_value=0, dtype=np.float32)

goodCount = 0

for i in range(rowCount):
    xData[i][0] = int(random.random() * 11 + 90)
    xData[i][1] = int(random.random() * 11 + 90)
    xData[i][2] = int(random.random() * 11 + 90)

    xALL = xData[i][0] * 0.6 + xData[i][1] * 0.3 + xData[i][2] * 0.1
    if xALL >= 95:
        yTrainData[i] = 1
        goodCount = goodCount + 1
    else:
        yTrainData[i] = 0

print("xData=%s" % xData)
print("yTrainData=%s" % yTrainData)
print("goodCount=%s" % goodCount)

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)
b = tf.Variable(80, dtype=tf.float32)

wn = tf.nn.softmax(w)

n1 = wn * x
n2 = tf.reduce_sum(n1) - b

y = tf.nn.sigmoid(n2)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2):
    for j in range(rowCount):
        result = sess.run([train, x, yTrain, wn, b, n2, y, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
        print(result)
