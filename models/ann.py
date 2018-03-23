"""
Set up a function class for feed-forward-neural-networks.

Compare with code on:
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/03_net.py

Parameters are:
    - units : units for hidden layer
    - learn : learn rate
    - epochs: number of training epochs
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
                predictions = tf.nn.sigmoid(out)
                auc, auc_op = tf.metrics.auc(labels=Y,
                                             predictions=predictions)
                tf.summary.scalar('auc_train', auc)

            with tf.name_scope('global_ops'):
                init = tf.global_variables_initializer()
                merged = tf.summary.merge_all()

        with tf.Session(graph=train_graph) as sess:
            init.run()
            # initialize local vars for auc metric.
            tf.initialize_local_variables().run()
            writer = tf.summary.FileWriter(logdir=logdir, graph=train_graph)
            for epoch in tqdm(range(epochs)):
                c, log, w_1, w_o = sess.run(fetches=[train, merged, w1, wo],
                                            feed_dict={X: self.data,
                                                       Y: self.labels})
                writer.add_summary(log, global_step=epoch)
            writer.close()

            self.w1, self.wo = w_1, w_o

    def predict(self, data, labels):
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
                    Y = tf.placeholder(shape=(None, 1),
                                       dtype=tf.float32,
                                       name='labels')
                with tf.name_scope('forward_pass'):
                    h1 = tf.nn.sigmoid(tf.matmul(X, w1))
                    predictions = tf.nn.sigmoid(tf.matmul(h1, wo))

                with tf.name_scope('summaries'):
                    auc, _ = tf.metrics.auc(labels=Y,
                                            predictions=predictions)

        with tf.Session(graph=val_graph) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            preds, auc = sess.run([predictions, auc],
                                  feed_dict={X: data,
                                             Y: labels})

        return preds, auc


class NN2L(object):
    def __init__(self, data, labels, structure):
        self.inp, self.h1, self.h2 = structure
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
                predictions = tf.nn.sigmoid(out)
                auc, auc_op = tf.metrics.auc(labels=Y,
                                             predictions=predictions)
                tf.summary.scalar('auc_train', auc)

            with tf.name_scope('global_ops'):
                init = tf.global_variables_initializer()
                merged = tf.summary.merge_all()

        with tf.Session(graph=train_graph) as sess:
            init.run()
            # initialize local vars for auc metric.
            tf.local_variables_initializer().run()
            writer = tf.summary.FileWriter(logdir=logdir, graph=train_graph)
            for epoch in tqdm(range(epochs)):
                c, log, w_1, w_2, w_o = \
                 sess.run([train, merged, w1, w2, wo],
                          feed_dict={X: self.data,
                                     Y: self.labels})
                writer.add_summary(log, global_step=epoch)
            writer.close()

            self.w1, self.w2, self.wo = w_1, w_2, w_o

    def predict(self, data, labels):
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
                    Y = tf.placeholder(shape=(None, 1),
                                       dtype=tf.float32,
                                       name='labels')
                with tf.name_scope('forward_pass'):
                    h1 = tf.nn.sigmoid(tf.matmul(X, w1))
                    h2 = tf.nn.sigmoid(tf.matmul(h1, w2))
                    predictions = tf.nn.sigmoid(tf.matmul(h2, wo))

                with tf.name_scope('summaries'):
                    auc, _ = tf.metrics.auc(labels=Y,
                                            predictions=predictions)

        with tf.Session(graph=val_graph) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            preds, auc = sess.run([predictions, auc],
                                  feed_dict={X: data,
                                             Y: labels})

        return preds, auc
