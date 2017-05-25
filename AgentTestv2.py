import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
from policyagentv2 import PolicyAgent


# Open environment
#env = gym.make('CartPole-v0') # Cartpole
#env = gym.make('MountainCar-v0')
#env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')

observation_space = env.observation_space
action_space = env.action_space

# Training Parameters
D = observation_space.shape[0]
H = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 5
GAMMA = 0.97
LOW_PASS_AMOUNT = 5
RENDER_FREQUENCY = 100
A = action_space.n


# Helper function to discount the reward
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Helper class to handle average reward over 100 episodes
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)
        if len(self.items) > 100:
            self.items.pop()

    def get_average(self):
        running_total = 0
        for i in range(len(self.items)):
            running_total += self.items[i]
        return running_total/100.

# Tensorflow setup
tf.reset_default_graph()
agent = PolicyAgent(s_size=D, a_size=A, h_size=H, lr=LEARNING_RATE)

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
running_reward_new = Queue()
reward_sum_new = 0

while episode_number < total_episodes:
    ep_history = []
    rendering = False
    if episode_number % RENDER_FREQUENCY == 0:
        rendering = True

    if rendering:
        env.render()

    # Calculate step
    x = np.reshape(observation, [1,D])
    a_dist = sess.run(agent.output, feed_dict={agent.state: x})
    action = np.random.choice(a_dist[0],p=a_dist[0])
    action = np.argmax(a_dist == action)


    # Take step
    observation1, reward, done, info = env.step(action)

    # Store info
    xs.append(x)
    ys.append(action)
    rs.append(reward)
    reward_sum += reward
    reward_sum_new += reward

    ep_history.append([observation, action, reward, observation1])
    observation = observation1

    if done:
        # Detect when we have solved the environment
        running_reward_new.enqueue(reward_sum_new)
        if running_reward_new.get_average() > 475:
            print "Environment solved in {} episodes.".format(episode_number)
            break

        ep_history = np.array(ep_history)
        ep_history[:,2] = discount_reward(ep_history[:,2])
        episode_number += 1

        # Append info for while episode
        epx = np.vstack(xs)
        epy = np.array(ys)
        epr = np.array(rs)

        # Discount reward
        discounted_epr = discount_reward(epr)

        # Standardize the discount reward
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # Calculate the new gradients, and store them in the buffer
        gradients = sess.run(agent.gradients, feed_dict={agent.reward_holder: discounted_epr, agent.action_holder: epy, agent.state: epx})
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

            # Reset data
            reward_sum = 0
            gradients_buffer = [np.zeros_like(k) for k in sess.run(agent.t_vars)]

        # Reset data
        observation = env.reset()
        xs, ys, rs = [], [], []
        reward_sum_new = 0


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



