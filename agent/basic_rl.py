import numpy as np
from agent.base_rl import BaseRL


class BasicRL(BaseRL):
    def __init__(self, n_actions=4, alpha=0.2, beta=0.1, q_init=0.0):
        super().__init__(n_actions=n_actions)
        self.alpha = alpha
        self.beta = beta
        self.q_init = q_init
        self.Q = np.full(n_actions, q_init, dtype=float)

    def get_action_probs(self):
        # p_t = softmax(beta * Q_t)
        return self.softmax(self.beta * self.Q)

    def choose_action(self):
        probs = self.get_action_probs()
        action = self.sample_action(probs)
        return action, probs

    def update(self, action, reward):
        # Q_{t+1}(a_t) = Q_t(a_t) + alpha * (r_t - Q_t(a_t))
        self.Q[action] = self.Q[action] + self.alpha * (reward - self.Q[action])