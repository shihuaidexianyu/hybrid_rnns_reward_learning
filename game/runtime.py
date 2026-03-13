import numpy as np


class DriftingBandit:
    def __init__(self, means=None, drift_std=2.0, reward_std=8.0):
        if means is None:
            means = [40, 55, 50, 30]
        self.means = np.array(means, dtype=float)
        self.drift_std = drift_std
        self.reward_std = reward_std

    def step(self, action):
        # 奖励围绕当前动作均值随机波动
        reward = np.random.normal(self.means[action], self.reward_std)

        # 所有动作的真实均值都发生一点漂移
        self.means += np.random.normal(0, self.drift_std, size=len(self.means))
        self.means = np.clip(self.means, 1, 100)

        return float(np.clip(reward, 1, 100))
