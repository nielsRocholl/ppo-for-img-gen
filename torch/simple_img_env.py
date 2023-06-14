import numpy as np
import torch
from gym.spaces import Discrete, Box
import gym
import time
from utils import visualize
from ppo_torch import Agent


class ImageEnv(gym.Env):
    def __init__(self, target_image):
        super(ImageEnv, self).__init__()
        # set the target image
        self.target_image = target_image
        # set the state
        self.state = np.zeros(self.target_image.shape)
        # set the state space
        self.state_space = self.target_image.shape
        # the state is a (9, 5, 5)
        self.observation_shape = target_image.shape
        # set the amount of frames to stack
        self.fps = 2
        # Define action space (0 = black, 1 = white)
        self.action_space = Discrete(2)
        # Define observation space
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.fps * 2 + 1, self.state_space[0], self.state_space[0]), dtype=np.uint8)
        self.history = []
        self.agent_position = 0
        self.agent_history = []
        self.reward_range = self.calculate_reward_range()
        self.converged = False

    def calculate_reward_range(self):
        white_pixels = len(np.where(self.target_image == 1)[0])
        black_pixels = len(np.where(self.target_image == 0)[0])
        reward_range = white_pixels * 1 + black_pixels * (white_pixels / black_pixels) + (self.state_space[0] * self.state_space[0]) * 0.1
        return reward_range

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state and the target image the reward is pixel based,
        it is important that the maximal amount of reward the agent can get by selecting white pixels is equal to the
        maximal amount of reward the agent can get by selecting black pixels. If this is not the case, the agent will
        always select the color that gives the most reward.
        :return: reward
        """
        flattened_state = self.state.flatten()
        flattened_target = self.target_image.flatten()
        # if the state is equal to the target image, return 5 (big reward)
        if np.array_equal(flattened_state, flattened_target):
            print("Target image reached!")
            self.converged = True
            return len(flattened_target) * 0.1
        # else if the pixel at the agent position is equal to the target pixel
        elif flattened_state[self.agent_position] == flattened_target[self.agent_position]:
            # if that pixel is white, return 1
            if flattened_state[self.agent_position] == 1:
                return 1
            # if that pixel is black, return 1 / (white_pixels / black_pixels)
            else:
                white_pixels = np.where(flattened_target == 1)[0]
                black_pixels = np.where(flattened_target == 0)[0]
                return len(white_pixels) / len(black_pixels)
        # else if the pixel at the agent position is not equal to the target pixel (bad choice)
        else:
            # prevent the agent from making the whole image black by giving a negative reward
            # when the agent makes a pixel that should be white black
            if flattened_state[self.agent_position] == 0 and flattened_target[self.agent_position] == 1:
                return -1
            else:
                white_pixels = np.where(flattened_target == 1)[0]
                black_pixels = np.where(flattened_target == 0)[0]
                return -(len(white_pixels) / len(black_pixels))

    def step(self, action):

        new_color = action
        # Convert the 1D position to 2D coordinates
        x = self.agent_position // self.state_space[1]
        y = self.agent_position % self.state_space[1]

        # Change the color of the selected pixel
        self.state[x][y] = new_color

        # append to history
        self.history.append(self.state)

        # Calculate the reward
        reward = self.calculate_reward()

        # move the agent
        if self.agent_position + 1 < self.state_space[0] * self.state_space[1]:
            self.agent_position += 1
            done = False
        else:
            if self.state.all() == 1.0 or self.state.all() == 0.0:
              reward = -(self.state_space[0] **2 * 0.1)
            else:
              percentage_matching = np.mean(self.state == self.target_image) * 100
              reward += percentage_matching
            done = True

        # create and state with the agent position
        agent_pos = np.zeros(self.state_space)
        agent_x = self.agent_position // self.state_space[1]
        agent_y = self.agent_position % self.state_space[1]
        agent_pos[agent_x][agent_y] = 1.0
        self.agent_history.append(agent_pos)
        # append agent position and reward_ranged target image to history
        state = np.concatenate(
            (self.history[-self.fps:], self.agent_history[-self.fps:], np.expand_dims(self.target_image, axis=0)),
            axis=0)
        return state, reward, done

    def reset(self):
        self.converged = False
        # Reset the environment to an all black image
        self.state = np.zeros(self.state_space)
        self.agent_position = 0
        agent_pos = np.zeros(self.state_space)
        agent_x = self.agent_position // self.state_space[1]
        agent_y = self.agent_position % self.state_space[1]
        agent_pos[agent_x][agent_y] = 1.0
        # reset history
        self.history = [np.zeros(self.state_space)] * self.fps
        self.agent_history = [agent_pos] * self.fps
        state = np.concatenate(
            (self.history[-self.fps:], self.agent_history[-self.fps:], np.expand_dims(self.target_image, axis=0)),
            axis=0)

        return state

    def get_num_actions(self):
        return self.state_space[0] * self.state_space[1]

    def get_target_image(self):
        return self.target_image

#
# from utils import load_image
#
# binary_image_array = load_image(kind="Hard", size=(28, 28))
# env = ImageEnv(binary_image_array)
# N = 20
# batch_size = 256
# n_epochs = 8
# alpha = 0.0005  # 0.0001 works well on 10x10
# agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
#               alpha=alpha, n_epochs=n_epochs,
#               input_dims=env.observation_space.shape)
# agent.load_models()
#
# obs = env.reset()
# agent.actor.eval()
#
# while True:
#     visualize(binary_image_array, obs[1], kind="Display")
#     action, prob, val = agent.choose_action(obs)
#     obs, reward, done = env.step(action)
#
#     if done:
#         break
