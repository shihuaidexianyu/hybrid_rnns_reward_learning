import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.base_agent import BaseAgent


class MemoryANN(BaseAgent, nn.Module):
    """
    Faithful-to-paper implementation sketch of Memory-ANN.

    Paper idea:
    - Reward module is an RNN-like module with hidden state s^(r)
    - Action-history module is another RNN-like module with hidden state s^(a)
    - Reward module updates only the chosen action's Q-value
    - Unchosen Q-values decay toward Q_init
    - Action-history module outputs the whole c vector
    - Choice logits = Q + c
    """

    def __init__(
        self,
        n_actions: int = 4,
        reward_state_dim: int = 32,
        action_state_dim: int = 32,
        forget_rate: float = 0.05,
        q_init: float = 0.0,
    ):
        BaseAgent.__init__(self, n_actions=n_actions)
        nn.Module.__init__(self)
        self.reward_state_dim = reward_state_dim
        self.action_state_dim = action_state_dim
        self.forget_rate = forget_rate
        self.q_init = q_init

        # Reward module:
        # input = [r_{t-1}, s^{(r)}_{t-1}]
        self.reward_fc1 = nn.Linear(1 + reward_state_dim, reward_state_dim)
        self.reward_fc2 = nn.Linear(reward_state_dim, 1)

        # Action-history module:
        # input = [onehot(a_{t-1}), s^{(a)}_{t-1}]
        self.action_fc1 = nn.Linear(n_actions + action_state_dim, action_state_dim)
        self.action_fc2 = nn.Linear(action_state_dim, n_actions)
        self.reset_state()

    def init_states(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        q0 = torch.full((batch_size, self.n_actions), self.q_init, device=device)
        c0 = torch.zeros(batch_size, self.n_actions, device=device)
        sr0 = torch.zeros(batch_size, self.reward_state_dim, device=device)
        sa0 = torch.zeros(batch_size, self.action_state_dim, device=device)
        return q0, c0, sr0, sa0

    def reset_state(self):
        self.q = None
        self.c = None
        self.sr = None
        self.sa = None
        self.prev_action = None
        self.prev_reward = 0.0

    def reward_module(self, r_prev, sr_prev):
        """
        r_prev: [B]
        sr_prev: [B, reward_state_dim]
        returns:
            sr_t: [B, reward_state_dim]
            q_new_chosen: [B, 1]
        """
        x = torch.cat([r_prev.unsqueeze(-1), sr_prev], dim=-1)
        sr_t = torch.tanh(self.reward_fc1(x))
        q_new_chosen = self.reward_fc2(sr_t)
        return sr_t, q_new_chosen

    def action_module(self, a_prev, sa_prev):
        """
        a_prev: [B] int64
        sa_prev: [B, action_state_dim]
        returns:
            sa_t: [B, action_state_dim]
            c_t: [B, n_actions]
        """
        a_onehot = F.one_hot(a_prev, num_classes=self.n_actions).float()
        x = torch.cat([a_onehot, sa_prev], dim=-1)
        sa_t = torch.tanh(self.action_fc1(x))
        c_t = self.action_fc2(sa_t)
        return sa_t, c_t

    def forward_step(self, q_prev, c_prev, sr_prev, sa_prev, a_prev, r_prev):
        """
        One trial transition:
        previous info -> current logits/probabilities
        """
        batch_size = q_prev.shape[0]
        a_onehot = F.one_hot(a_prev, num_classes=self.n_actions).float()

        # Reward module
        sr_t, q_new_chosen = self.reward_module(r_prev, sr_prev)

        # Forgetting on all Q-values first
        q_decay = q_prev + self.forget_rate * (self.q_init - q_prev)

        # Replace chosen action only
        q_t = q_decay * (1.0 - a_onehot) + q_new_chosen * a_onehot

        # Action-history module
        sa_t, c_t = self.action_module(a_prev, sa_prev)

        # Choice logits and probabilities
        logits = q_t + c_t
        probs = F.softmax(logits, dim=-1)

        return q_t, c_t, sr_t, sa_t, logits, probs

    def forward_sequence(self, actions, rewards):
        """
        actions: [B, T]  human actions (used teacher-forcing style)
        rewards: [B, T]  observed rewards
        Returns logits for predicting action at each t >= 1 using info from t-1.
        """
        B, T = actions.shape
        q_t, c_t, sr_t, sa_t = self.init_states(B, actions.device)

        logits_list = []

        # Use previous trial info to predict current action
        for t in range(1, T):
            a_prev = actions[:, t - 1]
            r_prev = rewards[:, t - 1]

            q_t, c_t, sr_t, sa_t, logits, probs = self.forward_step(
                q_t, c_t, sr_t, sa_t, a_prev, r_prev
            )
            logits_list.append(logits)

        return torch.stack(logits_list, dim=1)  # [B, T-1, n_actions]

    def get_action_probs(self):
        device = self.reward_fc1.weight.device

        if self.q is None or self.c is None or self.sr is None or self.sa is None:
            q_prev, c_prev, sr_prev, sa_prev = self.init_states(1, device=device)
        else:
            q_prev = self.q.to(device)
            c_prev = self.c.to(device)
            sr_prev = self.sr.to(device)
            sa_prev = self.sa.to(device)

        if self.prev_action is None:
            logits = q_prev + c_prev
            probs = F.softmax(logits, dim=-1)
            return probs.squeeze(0), q_prev, c_prev, sr_prev, sa_prev

        a_prev = torch.tensor([self.prev_action], dtype=torch.long, device=device)
        r_prev = torch.tensor([self.prev_reward], dtype=torch.float32, device=device)
        q_t, c_t, sr_t, sa_t, _, probs = self.forward_step(
            q_prev, c_prev, sr_prev, sa_prev, a_prev, r_prev
        )
        return probs.squeeze(0), q_t, c_t, sr_t, sa_t

    def choose_action(self):
        probs, next_q, next_c, next_sr, next_sa = self.get_action_probs()
        probs_np = probs.detach().cpu().numpy()
        action = self.sample_action(probs_np)
        self.q = next_q.detach()
        self.c = next_c.detach()
        self.sr = next_sr.detach()
        self.sa = next_sa.detach()
        return action, probs_np

    def update(self, action, reward):
        self.prev_action = action
        self.prev_reward = reward
