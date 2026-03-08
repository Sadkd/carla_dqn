# dqn_agent_dqn.py
# Standard Deep Q-Network (DQN) implementation
# Identical architecture and hyperparameters to dqn_agent.py (Double DQN)
# The ONLY difference is in train_step():
#   DDQN: policy net selects action, target net evaluates it (decoupled)
#   DQN:  target net both selects AND evaluates the best next action (coupled)
# All other components — network, buffer, optimizer, scheduler — are unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ==============================================
# Dual-Stream DQN Network
# Identical to DDQN — architecture is not what
# differs between DQN and Double DQN
# ==============================================
class DQNNet(nn.Module):
    def __init__(self, input_channels=16, num_actions=5, vector_size=2):
        super(DQNNet, self).__init__()

        # Visual stream with BatchNorm
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Vector stream
        self.vector_fc = nn.Sequential(
            nn.Linear(vector_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Combined action head
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, visual, vector):
        if visual.ndim == 3:
            visual = visual.unsqueeze(0)
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)

        visual_features = self.conv(visual)
        vector_features = self.vector_fc(vector)
        combined        = torch.cat([visual_features, vector_features], dim=1)
        return self.fc(combined)


# ==============================================
# N-Step Return Buffer
# Identical to DDQN
# ==============================================
class NStepBuffer:
    def __init__(self, n_steps=3, gamma=0.99):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.buffer  = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self):
        return len(self.buffer) == self.n_steps

    def get(self):
        n_step_reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r

        state_0,  action_0, _, _,           _ = self.buffer[0]
        _,        _,        _, next_state_n, _ = self.buffer[-1]
        any_done = any(d for _, _, _, _, d in self.buffer)

        return state_0, action_0, n_step_reward, next_state_n, any_done

    def clear(self):
        self.buffer.clear()


# ==============================================
# Standard DQN Agent
# ==============================================
class DQNAgent:
    def __init__(self, device='cpu', frame_stack=4, num_actions=5,
                 target_update_freq=2000, n_steps=3, gamma=0.99,
                 total_train_steps=1200000):
        self.device             = device
        self.frame_stack        = frame_stack
        self.num_actions        = num_actions
        self.target_update_freq = target_update_freq
        self.train_steps        = 0
        self.gamma              = gamma
        self.n_steps            = n_steps

        # input_channels = (3 RGB + 1 seg) * stack_size = 16
        input_channels = 4 * frame_stack

        # Policy network
        self.policy_net = DQNNet(
            input_channels=input_channels,
            num_actions=num_actions,
            vector_size=2
        ).to(self.device)

        # Target network
        self.target_net = DQNNet(
            input_channels=input_channels,
            num_actions=num_actions,
            vector_size=2
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=1e-4
        )

        # CosineAnnealingLR — identical to DDQN
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_train_steps,
            eta_min=1e-5
        )

        self.epsilon = 1.0

        # N-step buffer — identical to DDQN
        self.n_step_buffer = NStepBuffer(n_steps=n_steps, gamma=gamma)

        # Pre-allocated GPU tensors for inference
        self._visual_tensor = torch.zeros(
            1, input_channels, 84, 84,
            dtype=torch.float32, device=self.device
        )
        self._vector_tensor = torch.zeros(
            1, 2,
            dtype=torch.float32, device=self.device
        )

    def _state_to_tensors(self, state):
        visual_np = state["visual"]  # (stack, H, W, 4)
        vector_np = state["vector"]  # (2,)

        # (stack, H, W, 4) → (stack*4, H, W)
        visual_np = visual_np.transpose(0, 3, 1, 2).reshape(-1, 84, 84)

        self._visual_tensor.copy_(
            torch.from_numpy(visual_np).float().unsqueeze(0)
        )
        self._vector_tensor.copy_(
            torch.from_numpy(vector_np).float().unsqueeze(0)
        )
        return self._visual_tensor, self._vector_tensor

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            visual_t, vector_t = self._state_to_tensors(state)
            q_values = self.policy_net(visual_t, vector_t)
        return q_values.argmax().item()

    def get_max_q(self, state):
        with torch.no_grad():
            visual_t, vector_t = self._state_to_tensors(state)
            q_values = self.policy_net(visual_t, vector_t)
        return q_values.max().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.add(state, action, reward, next_state, done)

        if done:
            self.n_step_buffer.clear()
            return None

        if self.n_step_buffer.is_ready():
            return self.n_step_buffer.get()

        return None

    def _prepare_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        def stack_visual(state_list):
            visuals = []
            for s in state_list:
                v = s["visual"].transpose(0, 3, 1, 2).reshape(-1, 84, 84)
                visuals.append(v)
            return np.stack(visuals)

        def stack_vector(state_list):
            return np.stack([s["vector"] for s in state_list])

        visual      = torch.from_numpy(stack_visual(states)).float().to(self.device, non_blocking=True)
        next_visual = torch.from_numpy(stack_visual(next_states)).float().to(self.device, non_blocking=True)
        vector      = torch.from_numpy(stack_vector(states)).float().to(self.device, non_blocking=True)
        next_vector = torch.from_numpy(stack_vector(next_states)).float().to(self.device, non_blocking=True)
        actions_t   = torch.tensor(actions, dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_t   = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t     = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        return (visual, vector, actions_t, rewards_t,
                next_visual, next_vector, dones_t)

    def train_step(self, batch, gamma=0.99):
        """
        Standard DQN target — the key difference from Double DQN:

        DDQN:
            next_actions = policy_net(s').argmax()       # policy net SELECTS
            next_q       = target_net(s')[next_actions]  # target net EVALUATES
            → decoupled: selection and evaluation use different networks
            → reduces overestimation bias

        DQN (this implementation):
            next_q = target_net(s').max()                # target net SELECTS AND EVALUATES
            → coupled: same network does both jobs
            → prone to overestimation because upward bias in selection
              compounds with upward bias in evaluation
        """
        (visual, vector, actions, rewards,
         next_visual, next_vector, dones) = self._prepare_batch(batch)

        # Current Q values — identical to DDQN
        current_q = self.policy_net(visual, vector).gather(1, actions).squeeze(1)

        # ── Standard DQN target ──────────────────────────────────
        # target net selects AND evaluates the best next action
        # this is the ONLY line that differs from Double DQN
        with torch.no_grad():
            next_q = self.target_net(next_visual, next_vector).max(1)[0]
        # ────────────────────────────────────────────────────────

        # n-step discount — identical to DDQN
        gamma_n  = gamma ** self.n_steps
        target_q = rewards + gamma_n * next_q * (1 - dones)

        # Huber loss — identical to DDQN
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — identical to DDQN
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Sync target network — identical to DDQN
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"[DQN] Target synced at step {self.train_steps} "
                f"| LR: {current_lr:.2e}"
            )

        return loss.item()