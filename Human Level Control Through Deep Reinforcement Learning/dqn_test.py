#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import gym
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
import tensorflow as tf
import tensorlayer as tl

tf.app.flags.DEFINE_integer('training_episodes', 10)
tf.app.flags.DEFINE_boolean('enable_env_monitor', False)
tf.app.flags.DEFINE_integer('minibatch_size', 32)
tf.app.flags.DEFINE_integer('replay_memory_capacity', 1000000)
tf.app.flags.DEFINE_integer('target_network_update_freq', 10000)
tf.app.flags.DEFINE_float('discount_factor', 0.99)
tf.app.flags.DEFINE_float('learning_rate', 0.00025)
tf.app.flags.DEFINE_float('gradient_momentum', 0.95)
tf.app.flags.DEFINE_float('initial_exploration', 1)
tf.app.flags.DEFINE_float('final_exploration', 0.1)
tf.app.flags.DEFINE_integer('final_exploration_frame', 1000000)

FLAGS = tf.app.flags.FLAGS


def get_exploration_rate(step):
    return np.interp(step,
                     (0, FLAGS.final_exploration_frame),
                     (FLAGS.initial_exploration, FLAGS.final_exploration))


class ReplayMemory(object):
    """Replay memory"""
    def __init__(self, capacity):
        self._capacity = capacity
        self._steps = []

    def add_step(self, observation, action, next_observation, reward, done):
        step = ReplayStep(observation, action, next_observation, reward, done)
        self._steps.append(step)

        while len(self._steps) > self._capacity:
            self._steps.pop(0)

    def get_random_steps(self, minibatch_size):
        return random.sample(self._steps, minibatch_size)


class ReplayStep(object):
    """Replay step"""
    def __init__(self, observation, action, next_observation, reward, done):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward
        self.done = done


class TensorLayerDQN(object):
    """Deep Q-Network implemented via TensorLayer"""
    def __init__(self, env, session):
        self._env = env
        self._session = session

        with tf.name_scope('input'):
            input_shape = [None]
            input_shape.extend(env.observation_space.shape)
            self._inputs = tf.placefolder(tf.float32, input_shape, 'input')
            output_shape = (None, env.action_space.n)
            self._outputs = tf.placefolder(tf.float32, output_shape, 'output')

        self._network = tl.layers.InputLayer(self._inputs,
                                             name='input_layer')
        self._network = tl.layers.DenseLayer(self._network,
                                             n_units=200,
                                             act=tf.nn.relu,
                                             name='full_connect_layer1')
        self._network = tl.layers.DenseLayer(self._network,
                                             n_units=env.action_space.n,
                                             name='output_layer')

        self._action_probabilities = tf.nn.softmax(self._network.output)

        actions = tf.placeholder(tf.int32, shape=(None))
        rewards = tf.placeholder(tf.float32, shape=(None))
        loss = tl.rein.cross_entropy_reward_loss(self._network.output, actions, rewards)
        self._train = tf.train.RMSPropOptimizer(FLAGS.learning_rate, FLAGS.gradient_momentum) \
            .minimize(loss)

    def get_best_action(self, observation):
        pass

    def train(self, batches):
        pass

    def update_target_params(self):
        pass


class KerasDQN(object):
    """Deep Q-Network implemented via Keras"""
    def __init__(self, env, session):
        self._env = env
        self._session = session
        self._model = Sequential([
            Dense(200, input_shape=env.observation_space.shape),
            Activation('relu'),
            Dense(env.action_space.n, 200),
            Activation('softmax'),
        ])
        self._model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy')

    def get_best_action(self, observation):
        actions = self._model.predict(np.array([observation]))
        return np.argmax(actions[0])

    def train(self, batches):
        pass

    def update_target_params(self):
        """It is not possible to clone a tensorflow graph"""
        pass


def main(_):
    # Initialize tensorflow session
    session = tf.Session()
    # Initialize gym enviroment
    env = gym.make('CartPole-v0')
    if FLAGS.enable_env_monitor:
        timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        replay_name = 'data/cartpole-v0-experiment-{}'.format(timestamp)
        env = wrappers.Monitor(env, replay_name)
    # Initialize replay memory
    memory = ReplayMemory(FLAGS.replay_memory_capacity)
    # Initialize action-value function Q
    q = KerasDQN(session)
    train_step = 1
    # For episode = 1, M do
    for epoch in range(FLAGS.training_episodes):
        # Initialize sequence s1
        observation = env.reset()
        done = False
        # For t = 1, T
        episode_step = 1
        while not done:
            env.render()
            # With probability epsilon select a random action
            exploration_rate = get_exploration_rate(train_step)
            if random.random() <= exploration_rate:
                action = env.action_space.sample()
            # otherwise select the action with best promise
            else:
                action = q.get_best_action(observation)
            # Execute selected action and observe reward and image
            next_observation, reward, done, info = env.step(action)
            # Store transition in memory
            memory.add_step(observation, action, next_observation, reward, done)
            # Sample random minibatch of transitions from memory
            minibatch = memory.get_random_steps(FLAGS.minibatch_size)
            # Perform a SGD step with respect to the network parameter
            q.train(minibatch)
            if train_step % FLAGS.target_network_update_freq == 0:
                q.update_target_params()
            # Print training status
            print('Episode {}, Step {}, Done {}'
                  .format(epoch, episode_step, done))
            observation = next_observation
            episode_step += 1
            train_step += 1
    session.close()


if __name__ == '__main__':
    tf.app.run()
