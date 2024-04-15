import gym
import torch
import numpy as np
from torch import nn
from chess import Chess

gym.register(
    id='Chess-v0',
    entry_point='chess:Chess'
)

class net(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()

        self.state_dim = state_dim
        self.action_n = action_n

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, self.action_n)
        )       

        self.optim = torch.optim.Adam(self.parameters(), lr=1.0e-2)

        self.softmax = nn.Softmax()

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.net(x)
        return x
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action
    
    def update_policy(self, trajectories):
        elite_states, elite_actions = [], []

        for trajectory in trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))

        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return None

def get_trajectory(env, agent, trajectory_len):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    state = env.reset()
    state = np.concatenate((state['agent'], state['target']))
    trajectory['states'].append(state)

    for _ in range(trajectory_len):

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _, info = env.step(action)
        state = np.concatenate((state['agent'], state['target']))
        trajectory['total_reward'] += reward

        if done:
            break

        trajectory['states'].append(state)
    
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantilize = np.quantile(total_rewards, q = q_param)
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantilize]



#env = gym.make('Chess-v0',render_mode='human')
env = gym.make('Chess-v0')

state_dim = 4
action_count = 8

agent = net(state_dim, action_count)
agent.train()

episode_n = 1000
trajectory_n = 20
trajectory_len = 100
q_param = 0.8


for _ in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(mean_total_reward)

    elite_trajectories = get_elite_trajectories(trajectories, q_param)

    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
