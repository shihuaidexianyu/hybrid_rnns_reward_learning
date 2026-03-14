import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.base_agent import BaseAgent


class RLANN(BaseAgent, nn.Module):
    def __init__(self, n_actions=4, hidden_size=16, q_init=0.5, forgetting=0.05):
        BaseAgent.__init__(self, n_actions=n_actions)
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.q_init = q_init
        self.forgetting = forgetting

        # Reward ANN: [Q_{t-1}(a), r_{t-1}] -> Q_t(a)
        self.reward_mlp = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

        # Action-History ANN: a_{t-1} -> c_t (one scalar per action)
        # 这里用 one-hot(action) 作为输入，工程上更自然
        self.action_mlp = nn.Sequential(
            nn.Linear(n_actions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )
        self.reset_state()

    def init_q(self, batch_size, device=None):
        return torch.full((batch_size, self.n_actions), self.q_init, device=device)

    def reset_state(self):
        self.q = None
        self.prev_action = None
        self.prev_reward = 0.0

    def step(self, q_prev, prev_action_idx, prev_reward):
        """
        q_prev:          [B, A]
        prev_action_idx: [B]   long tensor, 上一轮选的动作编号
        prev_reward:     [B]   float tensor, 上一轮奖励

        return:
            q_new:   [B, A]
            c_t:     [B, A]
            logits:  [B, A]
            probs:   [B, A]
        """
        B, A = q_prev.shape
        device = q_prev.device

        # -------- 1) forgetting on all actions --------
        q_decay = (1.0 - self.forgetting) * q_prev + self.forgetting * self.q_init

        # -------- 2) Reward ANN updates chosen action only --------
        chosen_q_prev = q_prev.gather(1, prev_action_idx.unsqueeze(1)).squeeze(1)  # [B]
        reward_input = torch.stack([chosen_q_prev, prev_reward], dim=1)  # [B, 2]
        chosen_q_new = self.reward_mlp(reward_input).squeeze(1)  # [B]

        q_new = q_decay.clone()
        q_new.scatter_(1, prev_action_idx.unsqueeze(1), chosen_q_new.unsqueeze(1))

        # -------- 3) Action-History ANN produces perseveration vector --------
        prev_action_onehot = F.one_hot(prev_action_idx, num_classes=A).float()  # [B, A]
        c_t = self.action_mlp(prev_action_onehot)  # [B, A]

        # -------- 4) choice logits and probabilities --------
        logits = q_new + c_t
        probs = F.softmax(logits, dim=-1)

        return q_new, c_t, logits, probs

    def get_action_probs(self):
        device = self.action_mlp[0].weight.device

        if self.q is None:
            q_prev = self.init_q(batch_size=1, device=device)
        else:
            q_prev = self.q.to(device)

        if self.prev_action is None:
            logits = q_prev
            probs = F.softmax(logits, dim=-1)
            return probs.squeeze(0), q_prev

        prev_action_idx = torch.tensor([self.prev_action], dtype=torch.long, device=device)
        prev_reward = torch.tensor([self.prev_reward], dtype=torch.float32, device=device)
        q_new, _, _, probs = self.step(q_prev, prev_action_idx, prev_reward)
        return probs.squeeze(0), q_new

    def choose_action(self):
        probs, next_q = self.get_action_probs()
        probs_np = probs.detach().cpu().numpy()
        action = self.sample_action(probs_np)
        self.q = next_q.detach()
        return action, probs_np

    def update(self, action, reward):
        self.prev_action = action
        self.prev_reward = reward
