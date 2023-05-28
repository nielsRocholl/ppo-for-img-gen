import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def load_image(kind="Simple", size=(10, 10)):
    if kind == "Hard":
        # Load the image
        img = Image.open("../digits/digit_3.png")

        # Convert the image to grayscale
        img = img.convert("L")

        # Resize the image to the required size
        if size != (28, 28):
            img = img.resize(size)

        # Convert the image data to a numpy array
        img_data = np.array(img)

        # Binarize the image data
        binary_image_array = (img_data > 0).astype(np.uint8)
    else:
        # Create a 5x5 array filled with zeros
        binary_image_array = np.zeros(size)
        # Set the third column to 1.0
        binary_image_array[:, 2] = 1.0
        # set the last row to 1.0
        binary_image_array[3, :] = 1.0

    return binary_image_array


def visualize(target, state, kind="Print"):
    def clear_console():
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            print("\n" * 100)

    # concatenate the images horizontally
    images = np.hstack((target, state))
    if kind == "Print":
        # clear_console()
        print(images)
    elif kind == "Display":
        # images = cv2.resize(images, (500, 250))

        images = np.where(images > 0, 1.0, 0)

        # display the concatenated images
        cv2.imshow('Target vs Current State', images)
        cv2.waitKey(1)