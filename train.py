import numpy as np
import torch
import os
from carla_env import CarlaEnv
from dqn_agent import DQNAgent

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

EPISODES = 1000
TARGET_UPDATE = 10
MAX_STEPS = 1000  # 🔥 Prevent runaway episodes

env = CarlaEnv()

state_shape = (4, 84, 84)
agent = DQNAgent(state_shape, num_actions=3)

rewards = []
collision_history = []

best_avg_reward = -float("inf")
patience = 20  # 🔥 Increased patience
patience_counter = 0

for episode in range(EPISODES):

    state = env.reset()
    done = False
    total_reward = 0
    episode_collision = False
    step_count = 0

    while not done and step_count < MAX_STEPS:

        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store((state, action, reward, next_state, done))
        agent.train_step()

        state = next_state
        total_reward += reward
        step_count += 1

        if info.get("collision", False):
            episode_collision = True

    rewards.append(total_reward)
    collision_history.append(1 if episode_collision else 0)

    print(f"Episode {episode} | Reward: {total_reward:.2f} | "
          f"Epsilon: {agent.epsilon:.3f} | Steps: {step_count}")

    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    # 🔥 Early stopping logic (less aggressive )
    if len(rewards) > 100:
        current_avg = np.mean(rewards[-100:])

        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

np.save("logs/rewards.npy", rewards)
np.save("logs/collision_rate.npy", collision_history)

torch.save({
    "model_state_dict": agent.policy_net.state_dict(),
}, "checkpoints/dqn_carla_final.pth")

env.close()
