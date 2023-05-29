import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from simple_img_env import ImageEnv
from utils import load_image, visualize


if __name__ == '__main__':
    binary_image_array = load_image(kind="Hard", size=(10, 10))
    env = ImageEnv(binary_image_array)
    N = 20
    batch_size = 256
    n_epochs = 8
    alpha = 0.0001 # 0.0001 works well on 10x10
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 1000

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        # concatenate the images horizontally
        images = np.hstack((binary_image_array, observation[2]))
        visualize(binary_image_array, observation[1], kind="Print")
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'max score ', env.reward_range,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
