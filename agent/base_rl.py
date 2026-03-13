from abc import ABC, abstractmethod

import numpy as np


class BaseRL(ABC):
    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def sample_action(self, probs):
        return np.random.choice(self.n_actions, p=probs)

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, action, reward):
        raise NotImplementedError
