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

# # Geron, Aurelien (2017), Hands-On Machine Learning with Scikit-Learn & TensorFlow

# # policy gradients

import gym
env = gym.make("CartPole-v0")
obs = env.reset()
print(obs)
#env.render()

# +
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(10000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
# -

import numpy as np
np.mean(totals), np.std(totals), np.min(totals), np.max(totals)

# +
import tensorflow as tf

tf.reset_default_graph()

n_inputs = 4
n_hidden = 4
n_ouputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation = tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_ouputs, activation = tf.nn.elu, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

init = tf.global_variables_initializer()

# +
y = 1. - tf.to_float(action)
learning_rate = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, variable in grads_and_vars]
# -

grads_and_vars

# +
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
    
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# +
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


# -

discount_rewards([10, 0 ,-50], discount_rate= 0.8)
discount_and_normalize_rewards([[10, 0 ,-50], [10, 20]], discount_rate = 0.8)

# +
n_iterations = 250
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_rate = 0.95

sess = tf.Session()
init.run(session = sess)
for iteration in range(n_iterations):
    all_rewards = []
    all_gradients = []
    for game in range(n_games_per_update):
        current_rewards = []
        current_gradients = []
        obs = env.reset()
        for step in range(n_max_steps):
            action_val, gradients_val = sess.run(
            [action, gradients],
            feed_dict = {X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            current_rewards.append(reward)
            current_gradients.append(gradients_val)
            if done:
                break
        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)
    all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
    feed_dict = {}
    for var_index, grad_placeholder in enumerate(gradient_placeholders):
        mean_gradients = np.mean(
            [reward * all_gradients[game_index][step][var_index]
                for game_index, rewards in enumerate(all_rewards)
                for step, reward in enumerate(rewards)],
            axis = 0)
        feed_dict[grad_placeholder] = mean_gradients
    sess.run(training_op, feed_dict = feed_dict)
# -

totals = []
for episode in range(5):
    episode_rewards = 0
    obs = env.reset()
    for step in range(10000):
        action_var, _ =  sess.run(
            [action, gradients],
            feed_dict = {X: obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_var[0][0])
        env.render()
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
