import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make('CartPole-v0')

H = 10
batch_size = 5
learning_rate = 1e-2
gamma = 0.99
D = 4

tf.reset_default_graph()

# Placeholders (All of these a variable length lists ([None, 1]), that are run through the function discount_rewards
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")  # Fake label (not entirely sure what this is, but is
# is defined as y = 1 if action == 0 else 0, so it is the opposite of the action. Is stored as a list, hence "None",
# so it takes a variable length list.
advantages = tf.placeholder(tf.float32, name="reward_signal")  # The rewards, also stored as a list.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")  # The observations, again a list.

# Weights, get variable so we can set xavier initializer and access it in tvars
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())

# Layers, first and only hidden uses relu, output uses sigmoid
layer1 = tf.nn.relu(tf.matmul(observations, W1))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# Model (This is where logits would typically be (tf.matmul(x, weights) + biases). This sends the weights in the
# direction of making actions that gave good advantage (reward over time) more likely, and actions that didn't less
# likely
tvars = tf.trainable_variables()  # the trainable variables
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y) * (input_y + probability))  # if input_y == 0, then
# return log(probability) otherwise return log(1-probability)
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)  # gradients for the trainable variables (W1 and W2)

# Optimizer
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam optimizer
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))  # update the gradients


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

reward_stuff = []

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        if reward_sum/batch_size > 200 or rendering == True:
            env.render()
            rendering = True
            break

        x = np.reshape(observation,[1,D])

        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum +=reward

        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [],[],[],[],[],[]

            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad:gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                reward_stuff.append(reward_sum/batch_size)
                print 'Average reward for episode %f. Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)

                if reward_sum/batch_size >= 200:
                    print "Task solved in ", episode_number,' episodes!'
                    break

                reward_sum = 0
            observation = env.reset()
print episode_number,' Episodes completed.'


def low_pass(in_array, strength):
    for i in range(len(in_array) - strength * 2):
        running_total = 0
        for j in range(strength + 1):
            running_total += in_array[i + j]

        in_array[i + strength] = running_total / (strength + 1)

low_pass(reward_stuff, 5)
plt.plot(reward_stuff)
plt.show()