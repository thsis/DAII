"""
Set up a function class for feed-forward-neural-networks.

Compare with code on:
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/03_net.py

Parameters are:
    - learn : learn rate
    - epochs: number of training epochs
    - units: tuple specifying the size of each layer (input, hidden1, ...)
"""
import tensorflow as tf
from tqdm import tqdm


class NN1L(object):
    def __init__(self, data, labels, units):
        self.units = units
        self.data = data
        self.labels = labels
        self.rows, self.cols = data.shape

    def train(self, epochs, learn, logdir="."):
        train_graph = tf.Graph()

        with train_graph.as_default():
            with tf.name_scope('inputs'):
                X = tf.placeholder(dtype=tf.float32,
                                   shape=self.data.shape,
                                   name='X')
                Y = tf.placeholder(dtype=tf.float32,
                                   shape=(self.labels.shape[0], 1),
                                   name='Y')
            with tf.name_scope('hidden_1'):
                w1 = tf.Variable(
                    tf.random_normal(shape=(self.cols, self.units)),
                    name='weights_h')
                h1 = tf.nn.sigmoid(tf.matmul(X, w1),
                                   name='sigmoid_nodes')
            with tf.name_scope('output'):
                wo = tf.Variable(tf.random_normal(shape=(self.units, 1)),
                                 name='weights_o')
                # We don't apply the sigmoid yet. This is done later
                # by the cost-function.
                out = tf.matmul(h1, wo, name='out')
            with tf.name_scope('update'):
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                            labels=Y),
                    name='cost_cross_entropy')
                train = tf.train.GradientDescentOptimizer(learn).minimize(cost)

            with tf.name_scope('summaries'):
                tf.summary.scalar('cross_entropy', cost)

            with tf.name_scope('global_ops'):
                init = tf.global_variables_initializer()
                merged = tf.summary.merge_all()

        with tf.Session(graph=train_graph) as sess:
            init.run()
            writer = tf.summary.FileWriter(logdir=logdir, graph=train_graph)
            for epoch in tqdm(range(epochs)):
                c, log, w_1, w_o = sess.run(fetches=[train, merged, w1, wo],
                                            feed_dict={X: self.data,
                                                       Y: self.labels})
                writer.add_summary(log, global_step=epoch)
            writer.close()

            self.w1, self.wo = w_1, w_o

    def predict(self, data):
        val_graph = tf.Graph()

        with val_graph.as_default():
            with tf.name_scope('validation'):
                with tf.name_scope('weights'):
                    w1 = tf.constant(self.w1, name='hidden_1')
                    wo = tf.constant(self.wo, name='output')

                with tf.name_scope('inputs'):
                    X = tf.placeholder(shape=(None, self.cols),
                                       dtype=tf.float32,
                                       name='features')

                with tf.name_scope('forward_pass'):
                    h1 = tf.nn.sigmoid(tf.matmul(X, w1))
                    predictions = tf.nn.sigmoid(tf.matmul(h1, wo))

        with tf.Session(graph=val_graph) as sess:
            tf.global_variables_initializer().run()

            preds = sess.run(predictions,
                             feed_dict={X: data})

        return preds


class NN2L(object):
    def __init__(self, data, labels, units):
        self.inp, self.h1, self.h2 = units
        self.data = data
        self.labels = labels

    def train(self, epochs, learn, logdir="."):
        train_graph = tf.Graph()

        with train_graph.as_default():
            with tf.name_scope('inputs'):
                X = tf.placeholder(dtype=tf.float32,
                                   shape=self.data.shape,
                                   name='X')
                Y = tf.placeholder(dtype=tf.float32,
                                   shape=(self.labels.shape[0], 1),
                                   name='Y')
            with tf.name_scope('hidden_1'):
                w1 = tf.Variable(
                    tf.random_normal(shape=(self.inp, self.h1)),
                    name='weights_h')
                h1 = tf.nn.sigmoid(tf.matmul(X, w1),
                                   name='sigmoid_nodes')
            with tf.name_scope('hidden_2'):
                w2 = tf.Variable(
                    tf.random_normal(shape=(self.h1, self.h2)),
                    name='weights_h')
                h2 = tf.nn.sigmoid(tf.matmul(h1, w2),
                                   name='sigmoid_nodes')
            with tf.name_scope('output'):
                wo = tf.Variable(tf.random_normal(shape=(self.h2, 1)),
                                 name='weights_o')
                # We don't apply the sigmoid yet. This is done later
                # by the cost-function.
                out = tf.matmul(h2, wo, name='out')
            with tf.name_scope('update'):
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                            labels=Y),
                    name='cost_cross_entropy')
                train = tf.train.GradientDescentOptimizer(learn).minimize(cost)

            with tf.name_scope('summaries'):
                tf.summary.scalar('cross_entropy', cost)

            with tf.name_scope('global_ops'):
                init = tf.global_variables_initializer()
                merged = tf.summary.merge_all()

        with tf.Session(graph=train_graph) as sess:
            init.run()
            writer = tf.summary.FileWriter(logdir=logdir, graph=train_graph)
            for epoch in tqdm(range(epochs)):
                c, log, w_1, w_2, w_o = \
                 sess.run([train, merged, w1, w2, wo],
                          feed_dict={X: self.data,
                                     Y: self.labels})
                writer.add_summary(log, global_step=epoch)
            writer.close()

            self.w1, self.w2, self.wo = w_1, w_2, w_o

    def predict(self, data):
        val_graph = tf.Graph()

        with val_graph.as_default():
            with tf.name_scope('validation'):
                with tf.name_scope('weights'):
                    w1 = tf.constant(self.w1, name='hidden_1')
                    w2 = tf.constant(self.w2, name='hidden_2')
                    wo = tf.constant(self.wo, name='output')

                with tf.name_scope('inputs'):
                    X = tf.placeholder(shape=(None, self.inp),
                                       dtype=tf.float32,
                                       name='features')
                with tf.name_scope('forward_pass'):
                    h1 = tf.nn.sigmoid(tf.matmul(X, w1))
                    h2 = tf.nn.sigmoid(tf.matmul(h1, w2))
                    predictions = tf.nn.sigmoid(tf.matmul(h2, wo))

        with tf.Session(graph=val_graph) as sess:
            tf.global_variables_initializer().run()

            preds = sess.run(predictions,
                             feed_dict={X: data})

        return preds


class NN5L(object):
    def __init__(self, data, labels, units):
        self.inp, self.h1, self.h2, self.h3, self.h4, self.h5 = units
        self.data = data
        self.labels = labels

    def train(self, epochs, learn, logdir="."):
        train_graph = tf.Graph()

        with train_graph.as_default():
            with tf.name_scope('inputs'):
                X = tf.placeholder(dtype=tf.float32,
                                   shape=self.data.shape,
                                   name='X')
                Y = tf.placeholder(dtype=tf.float32,
                                   shape=(self.labels.shape[0], 1),
                                   name='Y')
            with tf.name_scope('hidden'):
                w1 = tf.Variable(
                    tf.random_normal(shape=(self.inp, self.h1)),
                    name='weights_h1')
                w2 = tf.Variable(
                    tf.random_normal(shape=(self.h1, self.h2)),
                    name='weights_h2')
                w3 = tf.Variable(
                    tf.random_normal(shape=(self.h2, self.h3)),
                    name='weights_h3')
                w4 = tf.Variable(
                    tf.random_normal(shape=(self.h3, self.h4)),
                    name='weights_h4')
                w5 = tf.Variable(
                    tf.random_normal(shape=(self.h4, self.h5)),
                    name='weights_h5')
                h1 = tf.nn.sigmoid(tf.matmul(X, w1),
                                   name='sigmoid_h1')
                h2 = tf.nn.sigmoid(tf.matmul(h1, w2),
                                   name='sigmoid_h2')
                h3 = tf.nn.sigmoid(tf.matmul(h2, w3),
                                   name='sigmoid_h3')
                h4 = tf.nn.sigmoid(tf.matmul(h3, w4),
                                   name='sigmoid_h4')
                h5 = tf.nn.sigmoid(tf.matmul(h4, w5),
                                   name='sigmoid_h5')

            with tf.name_scope('output'):
                wo = tf.Variable(tf.random_normal(shape=(self.h5, 1)),
                                 name='weights_o')
                # We don't apply the sigmoid yet. This is done later
                # by the cost-function.
                out = tf.matmul(h5, wo, name='out')
            with tf.name_scope('update'):
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=out,
                                                            labels=Y),
                    name='cost_cross_entropy')
                train = tf.train.GradientDescentOptimizer(learn).minimize(cost)

            with tf.name_scope('summaries'):
                tf.summary.scalar('cross_entropy', cost)

            with tf.name_scope('global_ops'):
                init = tf.global_variables_initializer()
                merged = tf.summary.merge_all()

        with tf.Session(graph=train_graph) as sess:
            init.run()
            writer = tf.summary.FileWriter(logdir=logdir, graph=train_graph)
            for epoch in tqdm(range(epochs)):
                c, log, w_1, w_2, w_3, w_4, w_5, w_o = \
                 sess.run([train, merged, w1, w2, w3, w4, w5, wo],
                          feed_dict={X: self.data,
                                     Y: self.labels})
                writer.add_summary(log, global_step=epoch)
            writer.close()

            self.w1, self.w2, self.w3, self.w4, self.w5, self.wo =\
                w_1, w_2, w_3, w_4, w_5, w_o

    def predict(self, data):
        val_graph = tf.Graph()

        with val_graph.as_default():
            with tf.name_scope('validation'):
                with tf.name_scope('weights'):
                    w1 = tf.constant(self.w1, name='hidden_1')
                    w2 = tf.constant(self.w2, name='hidden_2')
                    w3 = tf.constant(self.w3, name='hidden_3')
                    w4 = tf.constant(self.w4, name='hidden_4')
                    w5 = tf.constant(self.w5, name='hidden_5')
                    wo = tf.constant(self.wo, name='output')

                with tf.name_scope('inputs'):
                    X = tf.placeholder(shape=(None, self.inp),
                                       dtype=tf.float32,
                                       name='features')
                with tf.name_scope('forward_pass'):
                    h1 = tf.nn.sigmoid(tf.matmul(X, w1))
                    h2 = tf.nn.sigmoid(tf.matmul(h1, w2))
                    h3 = tf.nn.sigmoid(tf.matmul(h2, w3))
                    h4 = tf.nn.sigmoid(tf.matmul(h3, w4))
                    h5 = tf.nn.sigmoid(tf.matmul(h4, w5))
                    predictions = tf.nn.sigmoid(tf.matmul(h5, wo))

        with tf.Session(graph=val_graph) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            preds = sess.run(predictions,
                             feed_dict={X: data})

        return preds
