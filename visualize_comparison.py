"""
visualize_comparison.py
Overlays DQN and DDQN training results side by side for direct comparison.

Usage:
    python visualize_comparison.py

Expects:
    logs/        — DDQN logs (from train.py)
    logs_dqn/    — DQN  logs (from train_dqn.py)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Config ───────────────────────────────────────────────────
DDQN_DIR = "logs"
DQN_DIR  = "logs_dqn"
OUTPUT   = "logs/comparison_dqn_vs_ddqn.png"

DDQN_COLOR = "steelblue"
DQN_COLOR  = "tomato"

def moving_average(data, window=50):
    return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]

def load(directory, filename):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return np.load(path)
    return None

def plot_metric(ax, ddqn_data, dqn_data, ylabel, title, window=50,
                hline=None, hline_label=None):
    episodes = range(len(ddqn_data)) if ddqn_data is not None else range(len(dqn_data))

    if ddqn_data is not None:
        ax.plot(ddqn_data, alpha=0.15, linewidth=0.5, color=DDQN_COLOR)
        ax.plot(moving_average(ddqn_data, window), linewidth=2,
                color=DDQN_COLOR, label=f"DDQN MA({window})")

    if dqn_data is not None:
        ax.plot(dqn_data, alpha=0.15, linewidth=0.5, color=DQN_COLOR)
        ax.plot(moving_average(dqn_data, window), linewidth=2,
                color=DQN_COLOR,  label=f"DQN  MA({window})")

    if hline is not None:
        ax.axhline(hline, color="blue", linestyle="--",
                   linewidth=1.5, label=hline_label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Load data ────────────────────────────────────────────────
print("=" * 60)
print("DQN vs DDQN — Comparison")
print("=" * 60)

ddqn_rewards   = load(DDQN_DIR, "rewards.npy")
ddqn_collision = load(DDQN_DIR, "collision_rate.npy")
ddqn_lengths   = load(DDQN_DIR, "episode_length.npy")
ddqn_speeds    = load(DDQN_DIR, "avg_speed.npy")
ddqn_lane      = load(DDQN_DIR, "avg_lane_dev.npy")
ddqn_loss      = load(DDQN_DIR, "avg_loss.npy")
ddqn_q         = load(DDQN_DIR, "mean_max_q.npy")

dqn_rewards    = load(DQN_DIR,  "rewards.npy")
dqn_collision  = load(DQN_DIR,  "collision_rate.npy")
dqn_lengths    = load(DQN_DIR,  "episode_length.npy")
dqn_speeds     = load(DQN_DIR,  "avg_speed.npy")
dqn_lane       = load(DQN_DIR,  "avg_lane_dev.npy")
dqn_loss       = load(DQN_DIR,  "avg_loss.npy")
dqn_q          = load(DQN_DIR,  "mean_max_q.npy")

if ddqn_rewards is None and dqn_rewards is None:
    print("No log files found. Run train.py and train_dqn.py first.")
    exit()

if ddqn_rewards is not None:
    print(f"DDQN: {len(ddqn_rewards)} episodes loaded from {DDQN_DIR}/")
if dqn_rewards is not None:
    print(f"DQN:  {len(dqn_rewards)} episodes loaded from {DQN_DIR}/")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(18, 22))
fig.suptitle("DQN vs Double DQN — Training Comparison\nCARLA Autonomous Driving",
             fontsize=16, fontweight="bold", y=0.99)

# 1. Reward progression
plot_metric(axes[0, 0], ddqn_rewards, dqn_rewards,
            "Total Reward", "Reward Progression")

# 2. Reward distribution
ax2 = axes[0, 1]
if ddqn_rewards is not None:
    ax2.hist(ddqn_rewards, bins=50, alpha=0.5, color=DDQN_COLOR,
             label=f"DDQN (mean={np.mean(ddqn_rewards):.0f})", edgecolor="none")
    ax2.axvline(np.mean(ddqn_rewards), color=DDQN_COLOR,
                linestyle="--", linewidth=2)
if dqn_rewards is not None:
    ax2.hist(dqn_rewards,  bins=50, alpha=0.5, color=DQN_COLOR,
             label=f"DQN  (mean={np.mean(dqn_rewards):.0f})",  edgecolor="none")
    ax2.axvline(np.mean(dqn_rewards), color=DQN_COLOR,
                linestyle="--", linewidth=2)
ax2.set_xlabel("Reward")
ax2.set_ylabel("Frequency")
ax2.set_title("Reward Distribution", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Episode length
plot_metric(axes[1, 0], ddqn_lengths, dqn_lengths,
            "Steps Survived", "Episode Length (Survival Time)")

# 4. Collision rate
plot_metric(axes[1, 1], ddqn_collision, dqn_collision,
            "Collisions", "Collision Rate")

# 5. Average speed
plot_metric(axes[2, 0], ddqn_speeds, dqn_speeds,
            "Speed (km/h)", "Average Speed per Episode",
            hline=20.0, hline_label="Target 20 km/h")

# 6. Lane deviation
plot_metric(axes[2, 1], ddqn_lane, dqn_lane,
            "Distance to Lane Center (m)", "Average Lane Deviation")

# 7. Training loss
plot_metric(axes[3, 0], ddqn_loss, dqn_loss,
            "Huber Loss", "Training Loss")

# 8. Mean max Q-value
plot_metric(axes[3, 1], ddqn_q, dqn_q,
            "Q-Value", "Mean Max Q-Value")

plt.tight_layout()
os.makedirs("logs", exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nComparison plot saved: {OUTPUT}")
plt.show()

# ── Summary table ────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Metric':<30} {'DDQN':>12} {'DQN':>12}")
print("=" * 60)

def fmt(data, fn):
    return f"{fn(data):.2f}" if data is not None else "N/A"

metrics = [
    ("Avg Reward (all)",         ddqn_rewards,   dqn_rewards,   np.mean),
    ("Avg Reward (last 100)",    ddqn_rewards,   dqn_rewards,   lambda x: np.mean(x[-100:])),
    ("Avg Survival (last 100)",  ddqn_lengths,   dqn_lengths,   lambda x: np.mean(x[-100:])),
    ("Avg Speed (last 100)",     ddqn_speeds,    dqn_speeds,    lambda x: np.mean(x[-100:])),
    ("Avg Lane Dev (last 100)",  ddqn_lane,      dqn_lane,      lambda x: np.mean(x[-100:])),
    ("Total Collisions",         ddqn_collision, dqn_collision, sum),
    ("Final Avg Loss",           ddqn_loss,      dqn_loss,      lambda x: np.mean(x[-100:])),
    ("Final Avg Max Q",          ddqn_q,         dqn_q,         lambda x: np.mean(x[-100:])),
]

for name, d_data, q_data, fn in metrics:
    print(f"{name:<30} {fmt(d_data, fn):>12} {fmt(q_data, fn):>12}")

print("=" * 60)