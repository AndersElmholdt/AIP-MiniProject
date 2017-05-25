import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
from policyagent import PolicyAgent


# Open environment
env = gym.make('CartPole-v0') # Cartpole
observation_space = env.observation_space
action_space = env.action_space

# Training Parameters
D = observation_space.shape[0]
H = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 5
GAMMA = 0.97
LOW_PASS_AMOUNT = 0
RENDER_FREQUENCY = 1000
A = action_space.n


# Helper function to discount the reward
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Tensorflow setup
tf.reset_default_graph()
agent = PolicyAgent(s_size=D, h_size=H, lr=LEARNING_RATE)

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
gradients_buffer = [np.zeros_like(k) for k in sess.run(agent.t_vars)]

while episode_number < total_episodes:
    rendering = False
    if episode_number % RENDER_FREQUENCY == 0:
        rendering = True

    if rendering:
        env.render()

    # Calculate step
    x = np.reshape(observation, [1,D])
    action_prob = sess.run(agent.output, feed_dict={agent.state: x})
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

        # Append info for while episode
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(rs)

        # Discount reward
        discounted_epr = discount_reward(epr)

        # Standardize the discount reward
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # Calculate the new gradients, and store them in the buffer
        gradients = sess.run(agent.gradients, feed_dict={agent.state: epx, agent.action_holder: epy, agent.reward_holder: discounted_epr})
        for i, gradient in enumerate(gradients):
            gradients_buffer[i] += gradient

        # When enough episodes has run, update the weights
        if episode_number % BATCH_SIZE == 0:
            # Update the batch
            sess.run(agent.update_batch, feed_dict={agent.w1_gradient: gradients_buffer[0], agent.w2_gradient: gradients_buffer[1]})

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
            gradients_buffer = [np.zeros_like(k) for k in sess.run(agent.t_vars)]

        # Reset data
        observation = env.reset()
        xs, ys, rs = [], [], []


# Helper function to perform a low_pass filter on an array
def low_pass(in_array, strength):
    for i in range(len(in_array) - strength * 2):
        running_total = 0
        for j in range(strength + 1):
            running_total += in_array[i + j]

        in_array[i + strength] = running_total / (strength + 1)

# Plot data
low_pass(br, LOW_PASS_AMOUNT)
plt.plot(br)
plt.show()

