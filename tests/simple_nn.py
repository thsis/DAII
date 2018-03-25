import tensorflow as tf
import pandas as pd
import os

from models import ann
from tqdm import tqdm

# Set up test data
data_path = os.path.join("data", "ratios.csv")
data = pd.read_csv(data_path, sep=';')
data.head(5)

X_data = data.iloc[:50, :28]
Y_data = data.iloc[:50, 28]
X_data.shape[1]

outcomes = Y_data.values.reshape((50, 1))

# Conserve memory.
del data


# Define graph to conceptualize.
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('inputs'):
        X = tf.placeholder(dtype=tf.float32,
                           shape=X_data.shape,
                           name='X')
        Y = tf.placeholder(dtype=tf.float32,
                           shape=(Y_data.shape[0], 1),
                           name='Y')
    with tf.name_scope('hidden_1'):
        w1 = tf.Variable(tf.random_normal(shape=(X_data.shape[1], 10)),
                         name='weights_h')
        h1 = tf.nn.sigmoid(tf.matmul(X, w1),
                           name='sigmoid_nodes')

    with tf.name_scope('output'):
        wo = tf.Variable(tf.random_normal(shape=(10, 1)),
                         name='weights_o')
        # We don't apply the sigmoid yet. This is done by the cost-function.
        out = tf.matmul(h1, wo, name='out')
    with tf.name_scope('update'):
        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                    labels=Y),
            name='cost_cross_entropy')
        train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    with tf.name_scope('global_ops'):
        init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    init.run()
    for epoch in tqdm(range(1)):
        c, _ = sess.run([h1, train], feed_dict={X: X_data.values,
                                                Y: outcomes})
        if epoch % 1000 == 0:
            print(c)
c.shape


test = ann.NN2L(data=X_data.values, labels=outcomes, structure=(28, 10, 10))
test.train(1000, 0.5, 'tests/test_NN2L')
test.predict(data=X_data.values, labels=outcomes)
outcomes


test5L = ann.NN5L(data=X_data.values,
                  labels=outcomes,
                  structure=(28, 70, 77, 70, 7, 7))
test5L.train(10000, 0.03, 'tests/test_NN5L')
test5L.predict(data=X_data.values, labels=outcomes)
