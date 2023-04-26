import tensorflow as tf
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

random.seed()

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)

wn = tf.nn.softmax(w)

n1 = wn * x
n2 = tf.reduce_sum(n1)
y = tf.nn.sigmoid(n2)
loss = tf.abs(yTrain - y)

optimizer = tf.train.RMSPropOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5):
    xData = [int(random.random() * 8 + 93), int(random.random() * 8 + 93), int(random.random() * 8 + 93)]
    xALL = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xALL >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print(result)
    xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
    xALL = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xALL >= 95:
        yTrainData = 1
    else:
        yTrainData = 0

    result = sess.run([train, x, yTrain, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print(result)
