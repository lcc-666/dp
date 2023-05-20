import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

roundT = 100
learnRateT = 0.001

argt = sys.argv[1:]
print("argt: %s" % argt)

for v in argt:
    if v.startswith("-round="):
        roundT = int(v[len("-round="):])
    if v.startswith("-learnrate="):
        learnRateT = float(v[len("-learnrate="):])

fileData = pd.read_csv('exchangeData.txt', dtype=np.float32, header=None)

wholeData = np.reshape(fileData.values, (-1))
print("wholeData: %s" % wholeData)

cellCount = 3
unitCount = 5

testData = wholeData[-cellCount:]
print("testData: %s\n" % testData)

rowCount = wholeData.shape[0] - cellCount
print("rowCount: %d\n" % rowCount)

xData = [wholeData[i:i + cellCount] for i in range(rowCount)]
yTrainData = [wholeData[i + cellCount] for i in range(rowCount)]

print("xData: %s\n" % xData)
print("yTrainData: %s\n" % yTrainData)

x = tf.placeholder(shape=[cellCount], dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

cellT = tf.nn.rnn_cell.BasicLSTMCell(unitCount)

initState = cellT.zero_state(1, dtype=tf.float32)

h, finalState = tf.nn.dynamic_rnn(cellT, tf.reshape(x, [1, cellCount, 1]), initial_state=initState, dtype=tf.float32)
hr = tf.reshape(h, [cellCount, unitCount])

w2 = tf.Variable(tf.random_normal([unitCount, 1]), dtype=tf.float32)
b2 = tf.Variable(0.0, dtype=tf.float32)

y = tf.reduce_sum(tf.matmul(hr, w2) + b2)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(learnRateT)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(roundT):
    lossSum = 0.0
    for j in range(rowCount):
        result = sess.run([train, x, yTrain, y, h, finalState, loss], feed_dict={x: xData[j], yTrain: yTrainData[j]})
        lossSum = lossSum + float(result[len(result) - 1])
        if j == (rowCount - 1):
            print("i: %d, x: %s, yTrain: %s, y: %s, h: %s, finalState: %s, loss: %s, avgLoss: %10.10f\n" % (i, result[1], result[2], result[3], result[4], result[5], result[6], (lossSum / rowCount)))

result = sess.run([x, y], feed_dict={x: testData})
print("x: %s, y: %s\n" % (result[0], result[1]))