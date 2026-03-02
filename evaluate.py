import torch
import numpy as np
from dqn_agent import DQNAgent
from carla_env import CarlaEnv

EPISODES = 10

def evaluate():
    env = CarlaEnv()
    agent = DQNAgent()

    agent.model.load_state_dict(torch.load("checkpoints/dqn_latest.pth"))
    agent.model.eval()

    agent.epsilon = 0.01   # almost no exploration

    all_rewards = []
    collisions = 0

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        collided = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

            if info.get("collision", False):
                collided = True

        if collided:
            collisions += 1

        all_rewards.append(total_reward)
        print(f"[EVAL] Episode {ep} | Reward {total_reward}")

    print("\n===== EVALUATION RESULTS =====")
    print("Avg Reward:", np.mean(all_rewards))
    print("Collision Rate:", collisions / EPISODES)

    env.close()

if __name__ == "__main__":
    evaluate()

