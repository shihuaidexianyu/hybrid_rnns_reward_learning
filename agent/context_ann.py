import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.base_agent import BaseAgent


class ContextANN(BaseAgent, nn.Module):
    def __init__(self, n_actions=4, hidden_dim=32, forget_rate=0.05, q_init=0.0):
        BaseAgent.__init__(self, n_actions=n_actions)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.forget_rate = forget_rate
        self.q_init = q_init

        # Reward module:
        # paper form: [Q_{t-1}(a_{t-1}), r_{t-1}, Q_{t-1}]
        # here dims = 1 + 1 + n_actions
        self.reward_fc1 = nn.Linear(1 + 1 + n_actions, hidden_dim)
        self.reward_fc2 = nn.Linear(hidden_dim, 1)

        # Action-history module:
        # paper form: [a_{t-1}, c_{t-1}]
        # here a_{t-1} is one-hot => dims = n_actions + n_actions
        self.action_fc1 = nn.Linear(n_actions + n_actions, hidden_dim)
        self.action_fc2 = nn.Linear(hidden_dim, n_actions)
        self.reset_state()

    def init_q(self, batch_size, device=None):
        return torch.full(
            (batch_size, self.n_actions),
            self.q_init,
            dtype=torch.float32,
            device=device,
        )

    def init_c(self, batch_size, device=None):
        return torch.zeros(batch_size, self.n_actions, dtype=torch.float32, device=device)

    def reset_state(self):
        self.q = None
        self.c = None
        self.prev_action = None
        self.prev_reward = 0.0

    def reward_module(self, q_prev, a_prev, r_prev):
        """
        q_prev: [B, 4]
        a_prev: [B] int64 in {0,1,2,3}
        r_prev: [B]
        returns: updated chosen value [B, 1]
        """
        a_onehot = F.one_hot(a_prev, num_classes=self.n_actions).float()
        q_chosen = (q_prev * a_onehot).sum(dim=-1, keepdim=True)  # Q_{t-1}(a_{t-1})

        x = torch.cat([q_chosen, r_prev.unsqueeze(-1), q_prev], dim=-1)
        h = torch.tanh(self.reward_fc1(x))
        q_new_chosen = self.reward_fc2(h)  # scalar for chosen action
        return q_new_chosen

    def action_module(self, c_prev, a_prev):
        """
        c_prev: [B, 4]
        a_prev: [B]
        returns: c_t [B, 4]
        """
        a_onehot = F.one_hot(a_prev, num_classes=self.n_actions).float()
        x = torch.cat([a_onehot, c_prev], dim=-1)
        h = torch.tanh(self.action_fc1(x))
        c_t = self.action_fc2(h)
        return c_t

    def forward_step(self, q_prev, c_prev, a_prev, r_prev):
        """
        Single trial update.
        Inputs correspond to previous-trial information used to predict current choice.
        """
        B = q_prev.shape[0]
        a_onehot = F.one_hot(a_prev, num_classes=self.n_actions).float()

        # 1) Reward module updates only the chosen action
        q_new_chosen = self.reward_module(q_prev, a_prev, r_prev)  # [B,1]

        # 2) Forgetting for unchosen actions (paper keeps RL-ANN forgetting)
        q_decay = q_prev + self.forget_rate * (self.q_init - q_prev)

        # Replace chosen action with newly computed value
        q_t = q_decay * (1.0 - a_onehot) + q_new_chosen * a_onehot

        # 3) Action-history module
        c_t = self.action_module(c_prev, a_prev)

        # 4) Choice distribution
        logits = q_t + c_t
        probs = F.softmax(logits, dim=-1)

        return q_t, c_t, logits, probs

    def get_action_probs(self):
        device = self.reward_fc1.weight.device

        if self.q is None:
            q_prev = self.init_q(batch_size=1, device=device)
        else:
            q_prev = self.q.to(device)

        if self.c is None:
            c_prev = self.init_c(batch_size=1, device=device)
        else:
            c_prev = self.c.to(device)

        if self.prev_action is None:
            logits = q_prev + c_prev
            probs = F.softmax(logits, dim=-1)
            return probs.squeeze(0), q_prev, c_prev

        a_prev = torch.tensor([self.prev_action], dtype=torch.long, device=device)
        r_prev = torch.tensor([self.prev_reward], dtype=torch.float32, device=device)
        q_t, c_t, _, probs = self.forward_step(q_prev, c_prev, a_prev, r_prev)
        return probs.squeeze(0), q_t, c_t

    def choose_action(self):
        probs, next_q, next_c = self.get_action_probs()
        probs_np = probs.detach().cpu().numpy()
        action = self.sample_action(probs_np)
        self.q = next_q.detach()
        self.c = next_c.detach()
        return action, probs_np

    def update(self, action, reward):
        self.prev_action = action
        self.prev_reward = reward
