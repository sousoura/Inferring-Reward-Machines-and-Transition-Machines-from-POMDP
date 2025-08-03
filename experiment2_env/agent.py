import numpy as np
from collections import deque

import random


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        action = self.action_space.sample()
        return action
