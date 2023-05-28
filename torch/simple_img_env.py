import numpy as np
from gym.spaces import Discrete, Box
import gym
import time


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
        self.fps = 3
        # Define action space (0 = black, 1 = white)
        self.action_space = Discrete(2)
        # Define observation space
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.fps * 2 + 1, self.state_space[0], self.state_space[0]), dtype=np.uint8)
        self.history = []
        self.agent_position = 0
        self.agent_history = []
        self.reward_range = self.calculate_reward_range()

    def calculate_reward_range(self):
        white_pixels = len(np.where(self.target_image == 1)[0])
        black_pixels = len(np.where(self.target_image == 0)[0])
        reward_range = white_pixels * 1 + black_pixels * (white_pixels / black_pixels)
        return reward_range

    def calculate_reward(self):
        flattened_state = self.state.flatten()
        flattened_target = self.target_image.flatten()

        # if the color of the pixel is the same as the target image, return 1, else return -1
        # if done:
        #     if np.array_equal(flattened_state,  flattened_target):
        #         return 5
        #     else:
        #         return -5
        # if np.array_equal(flattened_state,  flattened_target):
        #     return 5
        if flattened_state[self.agent_position] == flattened_target[self.agent_position]:
            if flattened_state[self.agent_position] == 1:
                return 1
            else:
                white_pixels = np.where(flattened_target == 1)[0]
                black_pixels = np.where(flattened_target == 0)[0]
                return len(white_pixels) / len(black_pixels)
        else:
            return -(1 / (self.state_space[0] * self.state_space[1]))

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
            done = True

        # create and state with the agent position
        agent_pos = np.zeros(self.state_space)
        agent_x = self.agent_position // self.state_space[1]
        agent_y = self.agent_position % self.state_space[1]
        agent_pos[agent_x][agent_y] = 1.0
        self.agent_history.append(agent_pos)
        # append agent position anreward_ranged target image to history
        state = np.concatenate(
            (self.history[-self.fps:], self.agent_history[-self.fps:], np.expand_dims(self.target_image, axis=0)),
            axis=0)
        return state, reward, done

    def reset(self):
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


# # Create a 5x5 array filled with zeros
# binary_image_array = np.zeros((5, 5))
#
# # Set the third column to 1.0
# binary_image_array[:, 2] = 1.0
#
# env = ImageEnv(binary_image_array)
#
# obs = env.reset()
#
# while True:
#     print(obs[8])
#     action = env.action_space.sample()
#     obs, reward, done = env.step(action)
#     time.sleep(0.5)
#     if done:
#         break
