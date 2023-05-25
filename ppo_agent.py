import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from PIL import Image
from image_environment import ImageEnv

sns.set()

DEVICE = torch.device("mps")


# Policy and value model
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size, grid_size=(28, 28)):
        super().__init__()

        self.grid_size = grid_size

        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.policy_pixel_layers = nn.Sequential(
            nn.Linear(64 * grid_size[0] * grid_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, grid_size[0] * grid_size[1])  # Selects a pixel
        )

        self.policy_color_layers = nn.Sequential(
            nn.Linear(64 * grid_size[0] * grid_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Changes color
        )

        self.value_layers = nn.Sequential(
            nn.Linear(64 * grid_size[0] * grid_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def value(self, obs):
        obs = obs.view(-1, 1, self.grid_size[0], self.grid_size[1])
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        obs = obs.view(-1, 1, self.grid_size[0], self.grid_size[1])
        z = self.shared_layers(obs)
        policy_pixel_logits = self.policy_pixel_layers(z)
        policy_color_logits = self.policy_color_layers(z)
        return policy_pixel_logits, policy_color_logits

    def forward(self, obs):
        obs = obs.view(-1, 1, self.grid_size[0], self.grid_size[1])
        z = self.shared_layers(obs)
        policy_pixel_logits = self.policy_pixel_layers(z)
        policy_color_logits = self.policy_color_layers(z)
        value = self.value_layers(z)
        return policy_pixel_logits, policy_color_logits, value


class PPOTrainer():
    def __init__(self,
                 actor_critic,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=80,
                 value_train_iters=80,
                 policy_lr=3e-4,
                 value_lr=1e-2):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        policy_params = list(self.ac.shared_layers.parameters()) + \
                        list(self.ac.policy_pixel_layers.parameters()) + \
                        list(self.ac.policy_color_layers.parameters())

        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

        value_params = list(self.ac.shared_layers.parameters()) + \
                       list(self.ac.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, acts_pixel, acts_color, gaes, old_log_probs_pixel, old_log_probs_color):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()

            new_logits_pixel, new_logits_color = self.ac.policy(obs)
            new_logits_pixel, new_logits_color = Categorical(logits=new_logits_pixel), Categorical(
                logits=new_logits_color)
            new_log_probs_pixel = new_logits_pixel.log_prob(acts_pixel)
            new_log_probs_color = new_logits_color.log_prob(acts_color)

            # calculate the policy ration for the pixels and colors
            policy_ratio_pixel = torch.exp(new_log_probs_pixel - old_log_probs_pixel)
            policy_ratio_color = torch.exp(new_log_probs_color - old_log_probs_color)

            # calculate the clipped ratios for the pixels and the colors
            clipped_ratio_pixel = policy_ratio_pixel.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            clipped_ratio_color = policy_ratio_color.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

            # calculate the losses for the pixels and the colors
            clipped_loss_pixel = clipped_ratio_pixel * gaes
            full_loss_pixel = policy_ratio_pixel * gaes
            policy_loss_pixel = -torch.min(full_loss_pixel, clipped_loss_pixel).mean()

            clipped_loss_color = clipped_ratio_color * gaes
            full_loss_color = policy_ratio_color * gaes
            policy_loss_color = -torch.min(full_loss_color, clipped_loss_color).mean()

            # sum the pixel and policy loss
            policy_loss = policy_loss_pixel + policy_loss_color

            policy_loss.backward()
            self.policy_optim.step()

            # calculate KL-divergence
            kl_div_pixel = (old_log_probs_pixel - new_log_probs_pixel).mean()
            kl_div_color = (old_log_probs_color - new_log_probs_color).mean()
            # todo: check if the max operator is a logical choice
            if max(kl_div_pixel, kl_div_color) >= self.target_kl_div:
                break

    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.ac.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])


def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout.
    Returns training data in the shape (n_steps, observation_shape)
    and the cumulative reward.
    """
    # Create data storage
    train_data = [[], [], [], [], [], [],
                  []]  # obs, pixel_act, color_act, reward, val, pixel_act_log_probs, color_act_log_probs
    obs = env.reset()

    ep_reward = 0
    for step in range(max_steps):
        visualize(obs, env.target_image)

        # extract the pixel and color logits, and the value from the model
        pixel_logits, color_logits, val = model(torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE))

        # sample an action from the pixel and color logits
        pixel_distribution = Categorical(logits=pixel_logits)
        color_distribution = Categorical(logits=color_logits)
        pixel, color = pixel_distribution.sample(), color_distribution.sample()

        pixel_act_log_prob = pixel_distribution.log_prob(pixel).item()
        color_act_log_prob = color_distribution.log_prob(color).item()

        val = val.item()

        # take a step in the environment
        next_obs, reward, done = env.step(action=(pixel.item(), color.item()), current_step=step)
        print(f"pixel: {pixel.item()}, color: {color.item()}, reward: {reward}, done: {done}")
        # store the data
        for i, item in enumerate((obs, pixel.item(), color.item(), reward, val, pixel_act_log_prob, color_act_log_prob)):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done:
            break

    # train_data = [np.array(x) for x in train_data]
    train_data = [np.array(x, dtype=np.int32 if i in [1, 2] else np.float32) for i, x in enumerate(train_data)]

    # Do train data filtering use rewards and values to calculate gaes
    train_data[3] = calculate_gaes(train_data[4], train_data[5])

    return train_data, ep_reward


def visualize(target, state):
    # concatenate the images horizontally
    images = np.hstack((target, state))

    # display the concatenated images
    cv2.imshow('Target vs Current State', images)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def main():
    image = Image.open("digits/digit_3.png")
    image_array = np.array(image)

    # Ensure the image is grayscale
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=2)

    # Binarize the image array
    binary_image_array = np.where(image_array > 0, 1.0, 0.0)
    env = ImageEnv(binary_image_array)
    model = ActorCriticNetwork(env.state_space, env.action_space)
    model = model.to(DEVICE)
    # Define training params
    n_episodes = 100
    print_freq = 1

    ppo = PPOTrainer(
        model,
        policy_lr=3e-4,
        value_lr=1e-3,
        target_kl_div=0.02,
        max_policy_train_iters=40,
        value_train_iters=40)

    # Training loop
    ep_rewards = []
    for episode_idx in range(n_episodes):
        # Perform rollout
        train_data, reward = rollout(model, env)
        ep_rewards.append(reward)

        # Shuffle
        permute_idxs = np.random.permutation(len(train_data[0]))

        # Policy data
        obs = torch.tensor(train_data[0][permute_idxs],
                           dtype=torch.float32, device=DEVICE)
        acts_pixel = torch.tensor(train_data[1][permute_idxs],
                                  dtype=torch.int32, device=DEVICE)
        acts_color = torch.tensor(train_data[2][permute_idxs],
                                  dtype=torch.int32, device=DEVICE)
        gaes = torch.tensor(train_data[4][permute_idxs],
                            dtype=torch.float32, device=DEVICE)
        act_log_probs_pixel = torch.tensor(train_data[5][permute_idxs],
                                           dtype=torch.float32, device=DEVICE)
        act_log_probs_color = torch.tensor(train_data[6][permute_idxs],
                                           dtype=torch.float32, device=DEVICE)

        # Value data
        returns = discount_rewards(train_data[3])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # Train model
        ppo.train_policy(obs, acts_pixel, acts_color, gaes, act_log_probs_pixel, act_log_probs_color)
        ppo.train_value(obs, returns)

        # if (episode_idx + 1) % print_freq == 0:
        print('Episode {} | Avg Reward {:.1f}'.format(
            episode_idx + 1, np.mean(ep_rewards[-print_freq:])))


if __name__ == '__main__':
    main()
    # test()
