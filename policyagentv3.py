import tensorflow as tf


class PolicyAgent():
    def __init__(self, s_size, a_size, h_size, lr):
        # Placeholders
        self.state = tf.placeholder(tf.float32, shape=[None, s_size])
        self.action_holder = tf.placeholder(tf.int32, shape=[None])
        self.reward_holder = tf.placeholder(tf.float32)

        # Layers
        w1 = tf.get_variable("w1", shape=[s_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
        hidden = tf.nn.relu(tf.matmul(self.state, w1))
        w2 = tf.get_variable("w2", shape=[h_size, a_size], initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.nn.softmax(tf.matmul(hidden, w2))
        self.predict = tf.argmax(self.output, 1)

        # Loss function
        indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), indexes)
        loss = -tf.reduce_mean(tf.log(responsible_outputs) * self.reward_holder)

        # Model
        self.t_vars = tf.trainable_variables()
        self.gradients = tf.gradients(loss, self.t_vars)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.w1_gradient = tf.placeholder(tf.float32)
        self.w2_gradient = tf.placeholder(tf.float32)
        batch_gradients = [self.w1_gradient, self.w2_gradient]
        self.update_batch = optimizer.apply_gradients(zip(batch_gradients, self.t_vars))
