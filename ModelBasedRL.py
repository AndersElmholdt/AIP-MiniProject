from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam
import gym
import numpy as np
import tensorflow as tf
from policyagentv2 import PolicyAgent
import time

env = gym.make('CartPole-v1')
observation = env.reset()
done = False

observation_space = env.observation_space
action_space = env.action_space

# Training Parameters
D = observation_space.shape[0]
H = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 4
GAMMA = 0.97
LOW_PASS_AMOUNT = 4
RENDER_FREQUENCY = 1000
TOTAL_EPISODES = 1000
A = action_space.n
REWARD_TO_BEAT = 475

# Tensorflow setup
tf.reset_default_graph()
agent = PolicyAgent(s_size=D, a_size=A, h_size=H, lr=LEARNING_RATE)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Keras setup
model = Sequential()
model.add(Dense(512, batch_input_shape=(None, 6)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('sigmoid'))

optimizer = adam(lr=0.5, decay=0.005)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model_done = Sequential()
"""model_done.add(Dense(512, batch_input_shape=(None,4)))
model_done.add(Activation('relu'))
model_done.add(Dense(512))
model_done.add(Activation('relu'))
model_done.add(Dense(2))
model_done.add(Activation('softmax'))"""
model_done.add(Dense(512, batch_input_shape=(None,4)))
model_done.add(Activation('relu'))
model_done.add(Dense(2))
model_done.add(Activation('softmax'))

from keras.optimizers import SGD
optimizer_done = SGD(lr=0.5, decay=0.005)
model_done.compile(loss='mean_squared_error', optimizer=optimizer_done, metrics=['accuracy'])

"""optimizer_done = adam(lr = 0.0001)
model_done.compile(loss='mean_squared_error', optimizer=optimizer_done, metrics=['accuracy'])"""

xs, ys = [], []
episode_number = 1
test = 1

# Train model
while episode_number < 100:
    # Pick random action
    action = 1 if np.random.uniform() > 0.5 else 0

    # Store observation together with action
    x = np.append(np.append(done, observation), action)
    xs.append(x)

    if done:
        observation = env.reset()
        episode_number = episode_number + 1
        done = False
    else:
        observation, reward, done, _ = env.step(action)

    test = test + 1

    y = np.append(observation, np.append(reward, done))
    ys.append(y)

from keras.utils import np_utils


xs = np.array(xs)
ys = np.array(ys)
tests = np_utils.to_categorical(ys[:,5], 2)
model_done.fit(xs[:,0:4], tests, batch_size=1, epochs=5)
model.fit(xs, ys, batch_size=1, epochs=5)


def step_model(state, done, action):
    prediction = model.predict(np.reshape(np.array(np.append(np.append(done, state), action)), (1, 6)))

    observation = prediction[0, 0:4]
    reward = prediction[0, 4]
    done = prediction[0, 5]

    done = np.argmax(model_done.predict(np.reshape(np.array(state), (1,4))))

    print done

    if done > 0.5:
        done = True
    else:
        done = False
    return observation, reward, done


# Helper function to discount the reward
def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

# If true, it uses e-greedy, otherwise it uses the boltzman approach
USE_E_GREEDY = False

# Parameters for e-greedy, does nothing if use_e_greedy is false
E = 0.9
E_DECAY = 0.001


# Helper class to handle average reward over max_items amount of episodes
class Queue:
    def __init__(self, max_items):
        self.items = []
        self.max_items = max_items

    def enqueue(self, item):
        self.items.insert(0, item)
        if len(self.items) > self.max_items:
            self.items.pop()

    def get_average(self):
        running_total = 0.0
        for i in range(len(self.items)):
            running_total += self.items[i]
        return running_total/self.max_items

# Initial values
xs, ys, rs, ar = [],[],[],[]
episode_number = 1
reward_sum = 0
running_reward = 0
episode_reward = 0
episode_rewards_queue = Queue(100)

# Prepare to start
observation = env.reset()
gradients_buffer = [np.zeros_like(k) for k in sess.run(agent.t_vars)]
start_time = time.time()

while episode_number < TOTAL_EPISODES:
    # Calculate q
    x = np.reshape(observation, [1,D])
    q = sess.run(agent.output, feed_dict={agent.state: x})

    # Calculate action
    if USE_E_GREEDY:
        action = np.argmax(q)
        if np.random.uniform() < E:
            action = env.action_space.sample()
    else:
        action = np.random.choice(q[0], p=q[0])
        action = np.argmax(q == action)

    # Take step
    observation, reward, done = step_model(observation, False, action)

    # Store info
    xs.append(x)
    ys.append(action)
    rs.append(reward)
    reward_sum += reward
    episode_reward += reward

    if done:
        # Detect when we have solved the environment
        episode_rewards_queue.enqueue(episode_reward)
        if episode_rewards_queue.get_average() > REWARD_TO_BEAT:
            print "Environment solved in {} episodes.".format(episode_number)
            break

        # Decay E
        if USE_E_GREEDY:
            E = E * (1 - E_DECAY)

        # Increment episode number
        episode_number += 1

        # Append episode reward to list
        ar.append(episode_reward)

        # Store info for episode
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

        # When enough episodes has run, update the network
        if episode_number % BATCH_SIZE == 0:
            # Update the batch
            sess.run(agent.update_batch, feed_dict={agent.w1_gradient: gradients_buffer[0], agent.w2_gradient: gradients_buffer[1]})

            # Update the running reward and print current status
            running_reward += reward_sum
            print "Average reward for current batch was {}. Total average reward is {}".format(reward_sum / BATCH_SIZE, running_reward / episode_number)

            # Reset data
            reward_sum = 0
            gradients_buffer = [np.zeros_like(k) for k in sess.run(agent.t_vars)]

        # Reset data
        observation = step_model(observation, True, 0 if np.random.uniform() > 0.5 else 1)
        xs, ys, rs = [], [], []
        episode_reward = 0

print "Done... Elapsed time: {}".format(time.time() - start_time)


# Helper function to perform a low_pass filter on an array
def low_pass(in_array, strength):
    for i in range(len(in_array) - strength * 2):
        running_total = 0
        for j in range(strength + 1):
            running_total += in_array[i + j]

        in_array[i + strength] = running_total / (strength + 1)

# Plot data
plt.subplot(1,2,1)
plt.plot(ar)
plt.subplot(1,2,2)
low_pass(ar, LOW_PASS_AMOUNT)
plt.plot(ar)
plt.show()
