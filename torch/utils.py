import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot():
    path = "performances"
    files = os.listdir(path)
    random_runs = []
    policy_runs = []
    plt.figure(figsize=(7, 5), dpi=400)

    for file in files:
        run = np.load(f"{path}/{file}")
        # remove outliers"
        run = run[run <= 320]
        if len(run) >= 180:
            run = run[0:180]
            # normalize between 0 and 1
            run = run / np.max(320)
            if "random" in file:
                random_runs.append(run)
            else:
                policy_runs.append(run)
            plt.plot(run, color="grey", alpha=0.8)
            print(len(run))
    mean_policy = np.mean(policy_runs, axis=0)
    mean_random = np.mean(random_runs, axis=0)
    plt.plot(mean_policy, label="Mean PPO Policy", color="darkorange")
    plt.plot(mean_random, label="Mean Random Policy", color="purple")
    # dense grid
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Normalized Reward", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.title("PPO Policy Performance", fontsize=18)

    # Adjust the legend with a smaller font size and position outside the plot
    plt.legend(fontsize="large", bbox_to_anchor=(1, 0.35), loc='right')
    # Adjust the position of the axes
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13, top=0.9)
    plt.savefig("plots/ppo_result.png")
    plt.show()

def load_image(kind="Simple", size=(10, 10)):
    if kind == "Hard":
        # Load the image
        img = Image.open("../digits/digit_7.png")

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


def create_video_from_images():
    import os

    # Define the directory where the images are located
    image_directory = 'states/obama/'

    # Get a list of all image files in the directory
    image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')]

    # Sort the list of image files based on the number in their filename
    image_files = sorted(image_files, key=lambda filename: int(filename.split('_')[1].split('.')[0]))

    # add the last file 20 times to make the video longer
    for i in range(20):
        image_files.append(image_files[-1])

    # Specify the upscale factor
    scale = 20

    # Read the first image to get the shape
    img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)  # Reading the image in grayscale mode
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)  # Upscaling

    height, width = img.shape
    size = (width, height)

    # Create a VideoWriter object
    out = cv2.VideoWriter('convergence-obama.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, isColor=False)

    for image in image_files:
        # Reading each image and writing it to the video frame
        video_frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Reading the image in grayscale mode
        video_frame = cv2.resize(video_frame, (video_frame.shape[1] * scale, video_frame.shape[0] * scale),
                                 interpolation=cv2.INTER_NEAREST)  # Upscaling
        out.write(video_frame)

    out.release()


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 110x150 pixels
    img = cv2.resize(img, (28, 30), interpolation=cv2.INTER_NEAREST)

    # make the image square by adding white pixels to the left and right side
    old_size = img.shape[:2] # old_size is in (height, width) format
    desired_size = max(old_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Apply binary thresholding
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite("pixel_art/pika_processed.png", img)


# create_video_from_images()
preprocess_image("../digits/digit_7.png")
# plot()