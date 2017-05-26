import tensorflow as tf


class PolicyAgent():
    def __init__(self, s_size, h_size, lr):
        # Placeholders
        self.state = tf.placeholder(tf.float32, shape=[None, s_size])
        self.action_holder = tf.placeholder(tf.float32)
        self.reward_holder = tf.placeholder(tf.float32)

        # Layers
        w1 = tf.get_variable("w1", shape=[s_size, h_size])
        hidden = tf.nn.relu(tf.matmul(self.state, w1))
        w2 = tf.get_variable("w2", shape=[h_size, 1])
        self.output = tf.nn.sigmoid(tf.matmul(hidden, w2))

        # Model
        self.t_vars = tf.trainable_variables()
        logits = tf.log(self.action_holder * (self.action_holder - self.output) + (1 - self.action_holder) * (self.action_holder + self.output))
        loss = -tf.reduce_sum(logits * self.reward_holder)
        self.gradients = tf.gradients(loss, self.t_vars)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.w1_gradient = tf.placeholder(tf.float32)
        self.w2_gradient = tf.placeholder(tf.float32)
        batch_gradients = [self.w1_gradient, self.w2_gradient]
        self.update_batch = optimizer.apply_gradients(zip(batch_gradients, self.t_vars))
