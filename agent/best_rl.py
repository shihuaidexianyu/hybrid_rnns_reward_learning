import numpy as np
from agent.base_rl import BaseRL


class BestRL(BaseRL):
    def __init__(
        self,
        n_actions=4,
        alpha=0.2,
        beta=0.15,
        forgetting=0.05,
        kappa=3.0,
        bias=0.0,
        q_init=50.0,
    ):
        super().__init__(n_actions=n_actions)
        self.alpha = alpha
        self.beta = beta
        self.forgetting = forgetting
        self.kappa = kappa
        self.bias = bias
        self.q_init = q_init
        self.Q = np.full(n_actions, q_init, dtype=float)
        self.prev_action = None

    def get_perseveration_vector(self):
        c = np.zeros(self.n_actions, dtype=float)
        if self.prev_action is not None:
            c[self.prev_action] = self.kappa  # 给上一次选中的动作加上一个惯性项
        return c

    def get_choice_probs(self):
        # h_t = Q_t + c_t
        c = self.get_perseveration_vector()
        h = self.Q + c
        # p_t = softmax(beta * h_t)
        p = self.softmax(self.beta * h)
        return p, h, c

    def choose_action(self):
        p, h, c = self.get_choice_probs()
        action = self.sample_action(p)
        return action, p, h, c

    def apply_forgetting(self):
        # Q(a) <- (1-f) * Q(a) + f * Q_init
        self.Q = (1 - self.forgetting) * self.Q + self.forgetting * self.q_init

    def update(self, action, reward):
        """
        更新流程
        1) 先让所有 Q 做一次 forgetting
        2) 再对本轮选中的动作做奖励更新
        3) 记录 prev_action，供下一轮 perseveration 使用
        """
        self.apply_forgetting()

        # Q(a_t) <- Q(a_t) + alpha * (r_t - Q(a_t)) + b
        self.Q[action] = (
            self.Q[action] + self.alpha * (reward - self.Q[action]) + self.bias
        )

        self.prev_action = action