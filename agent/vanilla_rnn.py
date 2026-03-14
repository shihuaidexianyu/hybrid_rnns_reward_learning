import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.base_agent import BaseAgent


class VanillaRNN(BaseAgent, nn.Module):
    def __init__(self, n_actions=4, hidden_size=32):
        BaseAgent.__init__(self, n_actions=n_actions)
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        # 标准 vanilla RNN 单元：s_t = tanh(W_x x_t + W_h s_{t-1} + b)
        self.rnn_cell = nn.RNNCell(
            input_size=n_actions + 1,  # one-hot action + scalar reward
            hidden_size=hidden_size,
            nonlinearity="tanh",
        )

        # 读出层：hidden state -> logits for each action
        self.readout = nn.Linear(hidden_size, n_actions)
        self.reset_state()

    def reset_state(self):
        self.hidden = None
        self.prev_action = None
        self.prev_reward = 0.0

    def forward(self, prev_actions_onehot, prev_rewards, h0=None):
        """
        prev_actions_onehot: [B, T, n_actions]
        prev_rewards:        [B, T, 1]
        返回:
            logits_seq: [B, T, n_actions]
            h_T:        [B, hidden_size]
        """
        B, T, _ = prev_actions_onehot.shape
        x_seq = torch.cat([prev_actions_onehot, prev_rewards], dim=-1)

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=x_seq.device)
        else:
            h = h0

        logits_list = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            h = self.rnn_cell(x_t, h)  # 更新隐藏状态
            logits_t = self.readout(h)  # 输出当前 trial 的 logits
            logits_list.append(logits_t)

        logits_seq = torch.stack(logits_list, dim=1)
        return logits_seq, h

    def get_action_probs(self):
        device = self.readout.weight.device
        prev_action = torch.zeros(1, self.n_actions, device=device)
        prev_reward = torch.tensor(
            [[self.prev_reward]], dtype=torch.float32, device=device
        )

        if self.prev_action is not None:
            prev_action[0, self.prev_action] = 1.0

        if self.hidden is None:
            hidden = torch.zeros(1, self.hidden_size, device=device)
        else:
            hidden = self.hidden.to(device)

        x_t = torch.cat([prev_action, prev_reward], dim=-1)
        next_hidden = self.rnn_cell(x_t, hidden)
        logits = self.readout(next_hidden)
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0), next_hidden

    def choose_action(self):
        probs, next_hidden = self.get_action_probs()
        probs_np = probs.detach().cpu().numpy()
        action = self.sample_action(probs_np)
        self.hidden = next_hidden.detach()
        return action, probs_np

    def update(self, action, reward):
        self.prev_action = action
        self.prev_reward = reward
