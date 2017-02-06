#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import gym
from gym import wrappers
import random
import tensorflow as tf
import tensorlayer as tl

tf.app.flags.DEFINE_integer('training_episodes', 10)
tf.app.flags.DEFINE_boolean('enable_env_monitor', False)
tf.app.flags.DEFINE_float('greedy_epsilon', 0.1)
tf.app.flags.DEFINE_float('discount_factor', 0.99)
tf.app.flags.DEFINE_integer('minibatch_size', 32)
tf.app.flags.DEFINE_integer('target_network_update_freq', 10000)

FLAGS = tf.app.flags.FLAGS


class ReplayMemory(object):
    """Replay memory"""

    def __init__(self, capacity):
        self._capacity = capacity
        self._steps = []

    def append(self, step):
        self._steps.append(step)

        while len(self._steps) > self._capacity:
            self._steps.pop(0)

    def get_random_minibatch(self, minibatch_size):
        return random.sample(self._steps, minibatch_size)


class ReplayStep(object):
    """Replay step"""
    def __init__(self, observation, action, next_observation, reward, done):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward
        self.done = done


class DQN(object):
    """Deep Q-Network"""


def main(_):
    # Initialize gym enviroment
    env = gym.make('CartPole-v0')
    if FLAGS.enable_env_monitor:
        timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        replay_name = 'data/cartpole-v0-experiment-{}'.format(timestamp)
        env = wrappers.Monitor(env, replay_name)
    # Initialize replay memory
    memory = ReplayMemory()
    # Initialize action-value function Q
    q = DQN()
    train_step = 1
    # For episode = 1, M do
    for epoch in range(FLAGS.training_episodes):
        # Initialize sequence s1
        env.reset()
        observation, reward, done, info = env.step(0)
        # For t = 1, T
        episode_step = 1
        while not done:
            env.render()
            # With probability epsilon select a random action
            if random.random() <= FLAGS.greedy_epsilon:
                action = env.action_space.sample()
            # otherwise select the action with best promise
            else:
                action = q.get_best_action()
            # Execute selected action and observe reward and image
            next_observation, reward, done, info = env.step(action)
            # Store transition in memory
            replay_step = ReplayStep(observation, action, next_observation,
                                     reward, done)
            memory.append(replay_step)
            # Sample random minibatch of transitions from memory
            minibatch = memory.get_random_minibatch(FLAGS.minibatch_size)
            # Perform a SGD with respect to the network parameter
            q.train(minibatch)
            if train_step % FLAGS.target_network_update_freq == 0:
                q.update()
            # Print training status
            print('Episode {}, Step {}, Done {}'
                  .format(epoch, episode_step, done))
            observation = next_observation
            episode_step += 1
            train_step += 1


if __name__ == '__main__':
    tf.app.run()
