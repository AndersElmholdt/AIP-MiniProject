import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
import tensorflow.contrib.slim as slim


# Open environment
env = gym.make('CartPole-v0') # Cartpole
observation_space = env.observation_space
action_space = env.action_space


# ========================
#   Training Parameters
# ========================
D = observation_space.shape[0]
H = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 5
GAMMA = 0.99
LOW_PASS_AMOUNT = 5
RENDER_FREQUENCY = 100
A = action_space.n


# ========================
#    Tensorflow Setup
# ========================
tf.reset_default_graph()

# Placeholders
actions = tf.placeholder(tf.float32)
rewards = tf.placeholder(tf.float32)
observations = tf.placeholder(tf.float32, [None, D])

# Layers
W1 = tf.get_variable("W1", [D, H])
input_layer = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", [H, 1])
output_layer = tf.nn.sigmoid(tf.matmul(input_layer, W2))
chosen_action = tf.argmax(output_layer, 1)

# Model
train_vars = tf.trainable_variables()
logits = tf.log(actions * (actions - output_layer) + (1 - actions) * (actions + output_layer))
loss = -tf.reduce_sum(logits * rewards)
new_gradients = tf.gradients(loss, train_vars)

# Optimizer
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
W1_gradient = tf.placeholder(tf.float32)
W2_gradient = tf.placeholder(tf.float32)
batch_gradients = [W1_gradient, W2_gradient]
update_gradients = adam.apply_gradients(zip(batch_gradients, train_vars))


# Initial values
xs, ys, rs, br = [],[],[],[]
total_episodes = 10000
episode_number = 1
reward_sum = 0
running_reward = 0

# Prepare to start
observation = env.reset()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
gradients_buffer = [np.zeros_like(k) for k in sess.run(train_vars)]


# Helper function to discount the reward
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Helper function to perform a low_pass filter on an array
def low_pass(in_array, strength):
    for i in range(len(in_array) - strength * 2):
        running_total = 0
        for j in range(strength + 1):
            running_total += in_array[i + j]

        in_array[i + strength] = running_total / (strength + 1)

# Run until max episode count
while episode_number < total_episodes:
    rendering = False
    if episode_number % RENDER_FREQUENCY == 0:
        rendering = True

    if rendering:
        env.render()

    # Calculate step
    x = np.reshape(observation, [1,D])
    action_prob = sess.run(output_layer, feed_dict={observations: x})
    action = 1 if np.random.uniform() > action_prob else 0

    # Take step
    observation, reward, done, info = env.step(action)

    # Store info
    xs.append(x)
    ys.append(action)
    rs.append(reward)
    reward_sum += reward

    if done:
        episode_number += 1

        # Append info for whole episode
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(rs)

        # Discount reward
        discounted_epr = discount_reward(epr)

        # Standardize the discount reward
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # Calculate the new gradients, and store them in the buffer
        trained_gradients = sess.run(new_gradients, feed_dict={observations: epx, actions: epy, rewards: discounted_epr})
        for i, gradient in enumerate(trained_gradients):
            gradients_buffer[i] += gradient

        # When enough episodes has run, update the algorithm
        if episode_number % BATCH_SIZE == 0:
            # Update the gradients (maybe call this update weights)
            sess.run(update_gradients, feed_dict={W1_gradient: gradients_buffer[0], W2_gradient: gradients_buffer[1]})

            # Update the running reward and print current status
            running_reward += reward_sum
            print "Average reward for current batch was {}. Total average reward is {}".format(reward_sum / BATCH_SIZE, running_reward / episode_number)

            # Append average batch reward to list
            br.append(reward_sum/BATCH_SIZE)

            # Detect when we have solved the environment
            if reward_sum/BATCH_SIZE >= 200:
                print "Environment solved in {} episodes.".format(episode_number)
                break

            # Reset data
            reward_sum = 0
            gradients_buffer = [np.zeros_like(k) for k in sess.run(train_vars)]

        # Reset data
        observation = env.reset()
        xs, ys, rs = [], [], []


# Plot data
low_pass(br, LOW_PASS_AMOUNT)
plt.plot(br)
plt.show()

