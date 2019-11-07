# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Import and def

import numpy as np
import gym
import random
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
from stable_baselines import DQN

from curvytronClientCode import client
from curvytronClientCode import curvytron

serveraddress = "localhost:8080/#"
env = curvytron.CurvytronEnv(server=serveraddress, 
                             room='test2', 
                             name='pink_boi2', 
                             color='#ff0090')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model.learn(total_timesteps=int(2e3))
model.save("curvytron")
