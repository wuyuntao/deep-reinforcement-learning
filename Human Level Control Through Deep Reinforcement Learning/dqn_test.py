#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import gym
from gym import wrappers
import numpy as np
import random
import tensorflow as tf

tf.app.flags.DEFINE_string('env_name', "Breakout-v0",
                           "env_name")
tf.app.flags.DEFINE_integer('training_episodes', 200,
                            "training_episodes")
tf.app.flags.DEFINE_boolean('enable_env_monitor', False,
                            "enable_env_monitor")
tf.app.flags.DEFINE_integer('minibatch_size', 32,
                            "minibatch_size")
tf.app.flags.DEFINE_integer('replay_memory_capacity', 1000000,
                            "replay_memory_capacity")
tf.app.flags.DEFINE_integer('target_network_update_freq', 10000,
                            "target_network_update_freq")
tf.app.flags.DEFINE_float('discount_factor', 0.99,
                          "discount_factor")
tf.app.flags.DEFINE_float('learning_rate', 0.00025,
                          "learning_rate")
tf.app.flags.DEFINE_float('gradient_momentum', 0.95,
                          "gradient_momentum")
tf.app.flags.DEFINE_float('initial_exploration', 1,
                          "initial_exploration")
tf.app.flags.DEFINE_float('final_exploration', 0.1,
                          "final_exploration")
tf.app.flags.DEFINE_integer('final_exploration_frame', 1000000,
                            "final_exploration_frame")

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

    def add(self, observation, action, next_observation, reward, done):
        step = ReplayStep(observation, action, next_observation, reward, done)
        self._steps.append(step)

        while len(self._steps) > self._capacity:
            self._steps.pop(0)

    def get_batch(self, minibatch_size):
        if len(self._steps) < minibatch_size:
            return None

        return random.sample(self._steps, minibatch_size)

    @property
    def size(self):
        return len(self._steps)


class ReplayStep(object):
    """Replay step"""
    def __init__(self, observation, action, next_observation, reward, done):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward
        self.done = done


class DQN(object):
    """Deep Q-Network implemented via TensorFlow"""
    def __init__(self, env, session):
        self._env = env
        self._session = session

        with tf.variable_scope('input0'):
            self._input0 = tf.placeholder(
                tf.float32, (None, 210, 160, 3), name="input_layer")

        with tf.variable_scope('conv1'):
            self._conv1 = tf.contrib.layers.conv2d(
                self._input0,
                num_outputs=32,
                kernel_size=[8, 8],
                stride=[4, 4],
                padding="VALID",
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.relu)

        with tf.variable_scope('conv2'):
            self._conv2 = tf.contrib.layers.conv2d(
                self._conv1,
                num_outputs=64,
                kernel_size=[4, 4],
                stride=[2, 2],
                padding="VALID",
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.relu)

        with tf.variable_scope('conv3'):
            self._conv3 = tf.contrib.layers.conv2d(
                self._conv2,
                num_outputs=64,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="VALID",
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.relu)
            self._conv3_flat = tf.reshape(self._conv3, [-1, 22528])

        with tf.variable_scope('dense4'):
            self._dense4 = tf.contrib.layers.fully_connected(
                self._conv3_flat,
                num_outputs=512,
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.relu)

        with tf.variable_scope('dense5'):
            self._dense5 = tf.contrib.layers.fully_connected(
                self._dense4,
                num_outputs=512,
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.relu)

        with tf.variable_scope('output6'):
            self._q = tf.contrib.layers.fully_connected(
                self._dense5,
                num_outputs=self._env.action_space.n,
                weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                biases_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                activation_fn=tf.nn.sigmoid)
            self._q_action = tf.argmax(self._q, 1, name="q_action")
            self._q_max = tf.reduce_max(self._q, name="q_max")

        with tf.variable_scope('target'):
            self._q_target = tf.placeholder(
                tf.float32, [1, self._env.action_space.n],
                name="q_target")
            self._loss = tf.reduce_sum(tf.square(self._q_target - self._q))

        with tf.variable_scope('optimizer'):
            self._optimizer = tf.train.RMSPropOptimizer(
                learning_rate=FLAGS.learning_rate,
                decay=FLAGS.discount_factor,
                momentum=FLAGS.gradient_momentum,
                epsilon=0.01).minimize(self._loss)

        session.run(tf.global_variables_initializer())

    def get_best_action(self, observation):
        action = self._session.run([self._q_action], {
            self._input0: self._preprocess_observation(observation)
        })
        return action[0]

    def train(self, batches):
        for step in batches:
            q = step.reward
            if not step.done:
                next_q_max = self._session.run(self._q_max, {
                    self._input0: self._preprocess_observation(
                        step.next_observation)
                    })
                q += FLAGS.discount_factor * next_q_max
            q_target = np.zeros([1, self._env.action_space.n], np.float32)
            q_target[0, step.action] = q

            self._session.run(self._optimizer, {
                self._input0: self._preprocess_observation(
                    step.observation),
                self._q_target: q_target,
            })

    def update_target_params(self):
        pass

    def _preprocess_observation(self, observation):
        return (observation,)


def main(_):
    tf.set_random_seed(1)
    # Initialize tensorflow session
    session = tf.Session()
    # Initialize gym enviroment
    env = gym.make(FLAGS.env_name)
    if FLAGS.enable_env_monitor:
        timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        replay_name = 'data/{}-experiment-{}'.format(FLAGS.env_name, timestamp)
        env = wrappers.Monitor(env, replay_name)
    # Initialize replay memory
    memory = ReplayMemory(FLAGS.replay_memory_capacity)
    # Initialize action-value function Q
    q = DQN(env, session)
    train_step = 1
    # For episode = 1, M do
    for epoch in range(FLAGS.training_episodes):
        # Initialize sequence s1
        observation = env.reset()
        score = 0.0
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
            # Add penalty for lost live
            last_lives = env.ale.lives()
            # Execute selected action and observe reward and image
            next_observation, reward, done, info = env.step(action)
            if reward > 0:
                score += reward
            if env.ale.lives() < last_lives:
                reward -= 100
            # Store transition in memory
            memory.add(observation, action, next_observation, reward, done)
            # Sample random minibatch of transitions from memory
            minibatch = memory.get_batch(FLAGS.minibatch_size)
            # Perform a SGD step with respect to the network parameter
            if minibatch is not None:
                q.train(minibatch)
            if train_step % FLAGS.target_network_update_freq == 0:
                q.update_target_params()
            # Print training status
            print('Episode {}, Step {}/{}, Done {}, Reward {}/{}, Memory {}'
                  .format(epoch, episode_step, train_step, done, reward, score,
                          memory.size))
            observation = next_observation
            episode_step += 1
            train_step += 1
    session.close()


if __name__ == '__main__':
    tf.app.run()
