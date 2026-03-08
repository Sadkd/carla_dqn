# train_dqn.py
# Standard DQN training script
# Identical to train.py (Double DQN) in every way except:
#   - imports DQNAgent from dqn_agent_dqn.py
#   - saves logs    to  logs_dqn/
#   - saves checkpoints to checkpoints_dqn/
# This ensures DQN and DDQN results are stored separately
# and can be compared directly using visualize_results.py

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import deque
from carla_env import CarlaEnv
from dqn_agent_dqn import DQNAgent      # ← only import differs from train.py

# ==============================================
# GPU Setup
# ==============================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available!")

device = torch.device("cuda")
cudnn.benchmark     = True
cudnn.deterministic = False

print("=" * 60)
print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ==============================================
# Hyperparameters — identical to DDQN train.py
# ==============================================
NUM_EPISODES       = 2000
MAX_STEPS          = 300
BATCH_SIZE         = 32
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY          = 0.998
FRAME_STACK        = 4
TARGET_UPDATE_FREQ = 2000
MIN_BUFFER_SIZE    = 1000
MAX_BUFFER_SIZE    = 100000
NUM_ACTIONS        = 5
TRAIN_FREQ         = 2
N_STEPS            = 3

# resume training at checkpoint ep1000
# Resume training
RESUME_FROM    = "checkpoints_dqn/dqn_carla_ep1000.pth"
RESUME_EPISODE = 1000

# Separate output directories so DQN results don't overwrite DDQN results
LOG_DIR        = "logs_dqn"
CHECKPOINT_DIR = "checkpoints_dqn"

# Total expected train steps — used for CosineAnnealingLR
TOTAL_TRAIN_STEPS = NUM_EPISODES * MAX_STEPS * TRAIN_FREQ  # 1,200,000

# ==============================================
# Initialize environment and agent
# ==============================================
env = CarlaEnv(stack_size=FRAME_STACK)
agent = DQNAgent(
    device=device,
    frame_stack=FRAME_STACK,
    num_actions=NUM_ACTIONS,
    target_update_freq=TARGET_UPDATE_FREQ,
    n_steps=N_STEPS,
    gamma=GAMMA,
    total_train_steps=TOTAL_TRAIN_STEPS
)

print(f"Algorithm:         Standard DQN")
print(f"Episodes:          {NUM_EPISODES} | Max Steps: {MAX_STEPS}")
print(f"Batch size:        {BATCH_SIZE}   | Train freq: {TRAIN_FREQ}x per step")
print(f"Epsilon:           {EPS_START} → {EPS_END} (decay={EPS_DECAY})")
print(f"Replay buffer:     {MAX_BUFFER_SIZE:,} | Min to train: {MIN_BUFFER_SIZE}")
print(f"N-step returns:    {N_STEPS} steps")
print(f"Target sync:       every {TARGET_UPDATE_FREQ} steps")
print(f"Total train steps: {TOTAL_TRAIN_STEPS:,}")
print(f"LR schedule:       CosineAnnealing 1e-4 → 1e-5 over {TOTAL_TRAIN_STEPS:,} steps")
print(f"Log dir:           {LOG_DIR}/")
print(f"Checkpoint dir:    {CHECKPOINT_DIR}/")
print("=" * 60)

# ── Resume from checkpoint ──────────────────────────────────
checkpoint = torch.load(RESUME_FROM, map_location=device)
agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
agent.target_net.load_state_dict(checkpoint["target_model_state_dict"])
agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
agent.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
agent.epsilon = checkpoint["epsilon"]
print(f"Resumed from:      {RESUME_FROM}")
print(f"  Saved episode:   {checkpoint.get('episode', '?')}")
print(f"  Epsilon:         {agent.epsilon:.4f}")
print("=" * 60)
# ────────────────────────────────────────────────────────────


# Replay buffer
replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# # Logs
# rewards_log        = []
# collision_log      = []
# episode_length_log = []
# avg_speed_log      = []
# avg_lane_dev_log   = []
# avg_loss_log       = []
# mean_max_q_log     = []
# epsilon_log        = []

# resume training
# Load existing logs up to resume point
def load_log(path, max_ep):
    if os.path.exists(path):
        data = list(np.load(path))
        return data[:max_ep]
    return []

rewards_log        = load_log(f"{LOG_DIR}/rewards.npy",        RESUME_EPISODE)
collision_log      = load_log(f"{LOG_DIR}/collision_rate.npy", RESUME_EPISODE)
episode_length_log = load_log(f"{LOG_DIR}/episode_length.npy", RESUME_EPISODE)
avg_speed_log      = load_log(f"{LOG_DIR}/avg_speed.npy",      RESUME_EPISODE)
avg_lane_dev_log   = load_log(f"{LOG_DIR}/avg_lane_dev.npy",   RESUME_EPISODE)
avg_loss_log       = load_log(f"{LOG_DIR}/avg_loss.npy",       RESUME_EPISODE)
mean_max_q_log     = load_log(f"{LOG_DIR}/mean_max_q.npy",     RESUME_EPISODE)
epsilon_log        = load_log(f"{LOG_DIR}/epsilon.npy",        RESUME_EPISODE)

print(f"Loaded {len(rewards_log)} episodes from existing logs")

# ==============================================
# Training loop — identical to train.py
# ==============================================
# for episode in range(NUM_EPISODES):                      # normal training
for episode in range(RESUME_EPISODE, NUM_EPISODES):        # to resume training
    state = env.reset()

    episode_reward = 0
    collisions     = 0
    total_speed    = 0.0
    total_lane_dev = 0.0
    total_loss     = 0.0
    train_count    = 0
    max_q_values   = []

    for step in range(MAX_STEPS):

        # Max Q tracking
        max_q = agent.get_max_q(state)
        max_q_values.append(max_q)

        # Select action
        action = agent.select_action(state)

        # Perform action
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        if reward < -10:
            collisions += 1

        total_speed    += info.get("speed",    0.0)
        total_lane_dev += info.get("lane_dev", 0.0)

        # N-step buffer
        transition = agent.store_transition(state, action, reward, next_state, done)
        if transition is not None:
            replay_buffer.append(transition)

        state = next_state

        # Train TRAIN_FREQ times per step
        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            for _ in range(TRAIN_FREQ):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                loss  = agent.train_step(batch, GAMMA)
                total_loss  += loss
                train_count += 1

        if done:
            break

    # Episode metrics
    ep_length  = step + 1
    avg_speed  = total_speed    / ep_length
    avg_lane   = total_lane_dev / ep_length
    avg_loss   = total_loss     / max(1, train_count)
    mean_max_q = float(np.mean(max_q_values)) if max_q_values else 0.0

    rewards_log.append(episode_reward)
    collision_log.append(collisions)
    episode_length_log.append(ep_length)
    avg_speed_log.append(avg_speed)
    avg_lane_dev_log.append(avg_lane)
    avg_loss_log.append(avg_loss)
    mean_max_q_log.append(mean_max_q)
    epsilon_log.append(agent.epsilon)

    print(
        f"Episode {episode:4d} | "
        f"Reward: {episode_reward:7.2f} | "
        f"Len: {ep_length:3d} | "
        f"Collision: {collisions} | "
        f"Speed: {avg_speed:5.2f} km/h | "
        f"LaneDev: {avg_lane:.3f} | "
        f"Loss: {avg_loss:.4f} | "
        f"MaxQ: {mean_max_q:6.2f} | "
        f"Eps: {agent.epsilon:.4f}"
    )

    agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)

    # Checkpoint every 200 episodes
    if (episode + 1) % 200 == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            "model_state_dict":        agent.policy_net.state_dict(),
            "target_model_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict":    agent.optimizer.state_dict(),
            "scheduler_state_dict":    agent.scheduler.state_dict(),
            "epsilon":                 agent.epsilon,
            "episode":                 episode,
        }, f"{CHECKPOINT_DIR}/dqn_carla_ep{episode+1}.pth")

        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved  = torch.cuda.memory_reserved(0)  / 1e9
        print(
            f"[CHECKPOINT] Saved ep{episode+1} | "
            f"GPU: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved"
        )

# ==============================================
# Save all logs to logs_dqn/
# ==============================================
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

np.save(f"{LOG_DIR}/rewards.npy",        rewards_log)
np.save(f"{LOG_DIR}/collision_rate.npy", collision_log)
np.save(f"{LOG_DIR}/episode_length.npy", episode_length_log)
np.save(f"{LOG_DIR}/avg_speed.npy",      avg_speed_log)
np.save(f"{LOG_DIR}/avg_lane_dev.npy",   avg_lane_dev_log)
np.save(f"{LOG_DIR}/avg_loss.npy",       avg_loss_log)
np.save(f"{LOG_DIR}/mean_max_q.npy",     mean_max_q_log)
np.save(f"{LOG_DIR}/epsilon.npy",        epsilon_log)

torch.save({
    "model_state_dict":        agent.policy_net.state_dict(),
    "target_model_state_dict": agent.target_net.state_dict(),
    "optimizer_state_dict":    agent.optimizer.state_dict(),
    "scheduler_state_dict":    agent.scheduler.state_dict(),
    "epsilon":                 agent.epsilon,
    "episode":                 NUM_EPISODES,
}, f"{CHECKPOINT_DIR}/dqn_carla_final.pth")

print("=" * 60)
print("Training complete! — Standard DQN")
print(f"Avg Reward (all):        {np.mean(rewards_log):.2f}")
print(f"Avg Reward (last 100):   {np.mean(rewards_log[-100:]):.2f}")
print(f"Avg Survival (last 100): {np.mean(episode_length_log[-100:]):.1f} steps")
print(f"Avg Speed (last 100):    {np.mean(avg_speed_log[-100:]):.2f} km/h")
print(f"Avg Lane Dev (last 100): {np.mean(avg_lane_dev_log[-100:]):.3f} m")
print(f"Total Collisions:        {sum(collision_log)}")
print(f"Final GPU Memory:        {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
print("=" * 60)

env.close()