import numpy as np
import PIL.Image as Image
from gym.spaces import Discrete, Tuple


class ImageEnv:
    def __init__(self, target_image):
        # set the target image
        self.target_image = target_image
        # the state is a 28x28 pixel image (binary values)
        self.state_space = (28, 28)
        # Select a pixel: This is a tuple of x, y coordinates, and a binary value (0 or 1)
        self.action_space = Tuple((Discrete(self.state_space[0] * self.state_space[1]), Discrete(2)))
        self.state = []
        self.max_steps = 1000
        self.done_threshold = 0  # this is ofcourse way too high

    def step(self, action: (int, int), current_step):
        # action is a tuple (pixel_position, new_color)
        pixel_position, new_color = action

        # Convert the 1D position to 2D coordinates
        x = pixel_position // self.state_space[1]
        y = pixel_position % self.state_space[1]

        # Change the color of the selected pixel
        self.state[x][y] = new_color

        # Compute the reward and done flag
        mse = np.mean((self.state - self.target_image) ** 2)
        reward = -mse
        done = mse < self.done_threshold or current_step >= self.max_steps

        return self.state, reward, done

    def reset(self) -> np.ndarray:
        # Reset the environment to an all black image
        self.state = np.zeros(self.state_space)
        return self.state

    def get_num_actions(self):
        return self.state_space[0] * self.state_space[1]

    def get_target_image(self):
        return self.target_image

# env = ImageEnv(np.array(Image.open("digits/digit_3.png")))
# print(env.action_space)
