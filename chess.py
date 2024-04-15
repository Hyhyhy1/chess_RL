import gym
import torch
import numpy as np
from torch import nn
from torch import optim
import random
import matplotlib

import gym
from gym import spaces
import pygame
import numpy as np

class Chess(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Discrete(self.size*self.size)

        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([2, -1]),
            1: np.array([2, 1]),
            2: np.array([1, 2]),
            3: np.array([-1, 2]),
            4: np.array([-2, 1]),
            5: np.array([-2, -1]),
            6: np.array([-1, -2]),
            7: np.array([1, -2]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.size*self._agent_location[0]+self._agent_location[1]
    
    def _get_info(self):
        return {}
    
    def get_field_size(self):
        return self.size * self.size

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._visited = []
        
        self._field = np.zeros((self.size, self.size))
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._field[self._agent_location] = 1

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        direction = self._action_to_direction[action]

        temp1 = np.clip(self._agent_location + direction, 0, self.size - 1)
        temp2 = self._agent_location + direction

        if np.array_equal(temp1, temp2):
            self._visited.append(self._agent_location)
            self._agent_location = self._agent_location + direction
            self._field[self._agent_location] = 1
#        else:
#            reward = -100
        temp = np.ones((self.size, self.size))
        terminated = np.array_equal(self._field, temp2)
        reward = 0 if terminated else -1  # Binary sparse rewards
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        for place in self._visited:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * place,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
