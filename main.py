import gym
import torch
import numpy as np
from torch import nn
from chess import Chess
import matplotlib.pyplot as plt

gym.register(
    id='Chess-v0',
    entry_point='chess:Chess'
)

def get_epsilon_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)
    state_n = env.observation_space.n
    action_n = env.action_space.n
    q_function = np.zeros((state_n, action_n))
    
    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset() 
        action = get_epsilon_action(q_function[state], epsilon, action_n)

        for _ in range(trajectory_len):

            next_state, reward, done, _, info = env.step(action)
            next_action = get_epsilon_action(q_function[next_state], epsilon, action_n)
            q_function[state][action] += alpha * (reward + gamma * q_function[next_state][next_action] - q_function[state][action])

            state = next_state
            action = next_action

            total_rewards[episode] += reward

            if done:
                print('done')
                break

    return total_rewards



if __name__ == "__main__":
    env = gym.make('Chess-v0',render_mode='human')
    #env = gym.make('Chess-v0')
    total_rewards = SARSA(env, episode_n=400, trajectory_len=400)
    plt.plot(total_rewards)
    plt.show()
