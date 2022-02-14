import gym
import numpy as np
import tensorflow as tf

from models.architecture.graph import PpoGraph
from models.architecture.policy_header import Binomial
from models.architecture.value_target import AccumForwardRewards

env = gym.envs.make("CartPole-v0")
obs_space = env.observation_space.shape[0]

minimum_batch_size = 400
gradient_steps = 10
training_level = 3
log_sample_rate = 10

val_acc = AccumForwardRewards(1)

rl_model = PpoGraph(observation_size=obs_space, net_arch=[4, 4], learning_rate=8.0e-3, value_coef=0.1, entropy_coef=0.0, regularizer="l1", reg_param=0.001, policy=Binomial(),
                    activation=tf.nn.relu, initializer=tf.initializers.orthogonal, clip_range=0.1, pre_training_learning_rate=0.01, action_bounds=[0, 1])

total_training_rewards = [0. for _ in range(training_level)]
update_id = 0

while np.mean(total_training_rewards[-training_level:]) < 200:
    batch_actions = []
    batch_values = []
    batch_logits = []
    batch_observations = []
    batch_rewards = []
    batch_dones = []
    batch_total_episode_rewards = []

    obs = env.reset()
    done = False
    episode_rewards = []

    finished_rendering_first_episode = False

    while True:
        if (not finished_rendering_first_episode) & (-update_id % log_sample_rate == 1):
            env.render()
            # pass

        batch_observations.append(obs.reshape(-1, obs_space))
        action, dist_params, neg_logit, value = (np.asscalar(x) for x in rl_model.predict(obs.reshape(-1, obs_space)))
        obs, reward, done, _ = env.step(int(round(action)))

        batch_actions.append(action)
        batch_values.append(value)
        batch_logits.append(neg_logit)
        batch_rewards.append(reward)
        batch_dones.append(done)
        episode_rewards.append(reward)

        if done:
            batch_total_episode_rewards.append(np.sum(episode_rewards))

            obs, done, episode_rewards, finished_rendering_first_episode = env.reset(), False, [], True

            if len(batch_observations) > minimum_batch_size:
                obs_arr = np.squeeze(batch_observations)
                act_arr = np.array(batch_actions).reshape(-1, 1)
                old_logit_array = np.array(batch_logits).reshape(-1, 1)
                val_target_arr = val_acc.calculate_new_values(batch_rewards, batch_dones, batch_values).reshape(-1, 1)
                adv_arr = val_target_arr - np.array(batch_values).reshape(-1, 1)

                for steps in range(gradient_steps):
                    _, _, value, reg_loss = rl_model.train_step(obs_arr, act_arr, old_logit_array, val_target_arr, adv_arr, additional_fetches=[rl_model.value, rl_model.reg_loss])
                if -update_id % log_sample_rate == 1:
                    print(np.concatenate([value, val_target_arr, adv_arr], axis=1))
                    print("reg_loss: {}".format(reg_loss))
                break

    print("update {}, rewards: {}".format(update_id, np.mean(batch_total_episode_rewards)))
    total_training_rewards.append(np.mean(batch_total_episode_rewards))
    update_id += 1

print("Training finished!")

obs = env.reset()
counter = 0
while True:
    env.render()
    action = rl_model.sess.run(rl_model.policy.det_action, {rl_model.observation_input: obs.reshape(1, -1)})
    obs, reward, done, _ = env.step(int(round(np.asscalar(action))))
    if done:
        env.reset()
    counter += 1
