# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # LunarLander

# ## Downloads and installs

# !pip -q install pyglet
# !pip -q install pyopengl
# !pip -q install pyvirtualdisplay
# !pip -q install gym[box2d]
# !apt-get -qq -y install xvfb freeglut3-dev ffmpeg> /dev/null

# ## Import and def

# +
import os

import matplotlib.pyplot as plt
import matplotlib.animation
# %matplotlib inline
from IPython import display
from IPython.display import HTML
from pyvirtualdisplay import Display

import gym
from stable_baselines import DQN

# +
display = Display(visible=0, size=(1024, 768))
display.start()

os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

addframes = lambda frames, env: frames.append(env.render(mode = 'rgb_array'))

def show_env(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animate = lambda i: patch.set_data(frames[i])
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
    return ani.to_jshtml()


# -

# ## Random

frames = []
env = gym.make("LunarLander-v2")
for i_episode in range(3):
    observation = env.reset()
    for t in range(100):
        addframes(frames, env)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
HTML(show_env(frames))

# ## Deep Q-Learning

# +
# This is example code from https://github.com/hill-a/stable-baselines
# -

# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("dqn_lunar_new")
del model  # delete trained model to demonstrate loading

# +
# Load the trained agent
model = DQN.load("dqn_lunar")

# Enjoy trained agent
obs = env.reset()
frames = []
for i in range(1000):
    addframes(frames, env)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        print("Episode finished after {} timesteps".format(t+1))
        break
HTML(show_env(frames))
