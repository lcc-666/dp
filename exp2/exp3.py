import tensorflow as tf
import numpy as np
import pandas as pd

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
roundCount = 100
learnRate = 0.01
argt = sys.argv[1:]
for v in argt:
    if v.startswith("-round="):
        roundCount = int(v[len("-round="):])
    if v.startswith("-learnrate="):
        learnRate = float(v[len("-learnrate="):])

fileData = pd.read_csv('checkData.txt', dtype=np.float32, header=None)
wholeData = fileData.values

#增加训练数据
# print(len(wholeData))
# for i in range(5):
#     wholeData = np.vstack((wholeData, wholeData))
# print(len(wholeData))

rowCount = wholeData.shape[0]
# print("wholeData=%s" % wholeData)
# print("rowCount=%d" % rowCount)
x = tf.placeholder(shape=[25], dtype=tf.float32)

yTrain = tf.placeholder(shape=[3], dtype=tf.float32)



#修改卷积核大小
filterlT = tf.Variable(tf.ones([3, 3, 1, 1]), dtype=tf.float32)
n1 = tf.nn.conv2d(input=tf.reshape(x, [1, 5, 5, 1]), filter=filterlT, strides=[1, 1, 1, 1], padding='SAME')

filter2T = tf.Variable(tf.ones([2, 2, 1, 1]), dtype=tf.float32)
n2 = tf.nn.conv2d(input=tf.reshape(n1, [1, 5, 5, 1]), filter=filter2T, strides=[1, 1, 1, 1], padding='VALID')

filter3T = tf.Variable(tf.ones([2, 2, 1, 1]), dtype=tf.float32)
n3 = tf.nn.conv2d(input=tf.reshape(n2, [1, 4, 4, 1]), filter=filter3T, strides=[1, 1, 1, 1], padding='VALID')

n3f = tf.reshape(n3, [1, 9])

w4 = tf.Variable(tf.random_normal([9, 16]), dtype=tf.float32)
b4 = tf.Variable(0, dtype=tf.float32)
n4 = tf.nn.tanh(tf.matmul(n3f, w4) + b4)
w5 = tf.Variable(tf.random_normal([16, 3]), dtype=tf.float32)
b5 = tf.Variable(0, dtype=tf.float32)
n5 = tf.reshape(tf.matmul(n4, w5) + b5, [-1])
y = tf.nn.softmax(n5)

loss = -tf.reduce_mean(yTrain * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
optimizer = tf.train.RMSPropOptimizer(learnRate)
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(roundCount):
    lossSum = 0.0
    for j in range(rowCount):
        result = sess.run([train, x, yTrain, y, loss], feed_dict={x: wholeData[j][0:25], yTrain: wholeData[j][25: 28]})
        lossT = float(result[len(result) - 1])
        lossSum = lossSum + lossT
        if j == (rowCount - 1):
            print("i:%d, loss:%10.10f, avgLoss:%10.10f" % (i, lossT, lossSum / (rowCount + 1)))

print(sess.run([y, loss], feed_dict={x: [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     yTrain: [1, 0, 0]}))

print(sess.run([y, loss], feed_dict={x: [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                                     yTrain: [0, 1, 0]}))
print(sess.run([y, loss], feed_dict={x: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                     yTrain: [0, 0, 1]}))
