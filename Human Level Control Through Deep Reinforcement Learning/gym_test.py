#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import gym
from gym import wrappers

timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
REPLAY = 'data/cartpole-v0-experiment-{}'.format(timestamp)

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, REPLAY)

print('action', env.action_space)
print('observation', env.observation_space)
print('observation high', env.observation_space.high)
print('observation low', env.observation_space.low)

env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print('observation', observation)
    print('reward', reward)
    print('done', done)
    print('info', info)
    env.render()

# Upload replay to openai
# API_KEY='sk_voxg5d6PQF2QbShtTfv1OQ'
# GITHUB_GIST='https://gist.github.com/gdb/b6365e79be6052e7531e7ba6ea8caf23'
# gym.upload(REPLAY, writeup=GITHUB_GIST api_key=API_KEY)
