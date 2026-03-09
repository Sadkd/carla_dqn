import torch
import numpy as np
import sys
import time
import carla

from dqn_agent import DQNAgent
from carla_env import CarlaEnv


EPISODES = 10
FRAME_STACK = 4
NUM_ACTIONS = 5


# =========================================================
# Spectator Camera
# =========================================================
def set_spectator(world, vehicle):

    transform = vehicle.get_transform()

    spectator_transform = carla.Transform(
        carla.Location(
            x=transform.location.x - 6.0 * transform.get_forward_vector().x,
            y=transform.location.y - 6.0 * transform.get_forward_vector().y,
            z=transform.location.z + 3.5
        ),
        carla.Rotation(
            pitch=-15,
            yaw=transform.rotation.yaw,
            roll=0
        )
    )

    world.get_spectator().set_transform(spectator_transform)


# =========================================================
# Evaluation
# =========================================================
def evaluate(checkpoint_path="checkpoints/ddqn_carla_final.pth", render=True):

    env = CarlaEnv(stack_size=FRAME_STACK)

    device = torch.device("cpu")

    agent = DQNAgent(
        device=device,
        frame_stack=FRAME_STACK,
        num_actions=NUM_ACTIONS
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_model_state_dict"])

    agent.policy_net.eval()

    # No exploration during evaluation
    agent.epsilon = 0.0

    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {checkpoint.get('episode', '?')} episodes")
    print(f"  Saved epsilon: {checkpoint.get('epsilon', '?'):.4f}")
    print(f"  Render mode: {'ON — watch in CARLA window' if render else 'OFF'}")
    print("=" * 60)

    world = env.world if render else None

    all_rewards = []
    all_lengths = []
    all_speeds = []
    all_lane_devs = []

    collisions = 0


    for ep in range(EPISODES):

        state = env.reset()

        done = False
        step = 0

        total_reward = 0
        total_speed = 0
        total_lane_dev = 0
        collided = False

        print(f"\n[EVAL] Episode {ep} starting...")

        while not done:

            if render and env.vehicle is not None:
                set_spectator(world, env.vehicle)

            # Select action
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_speed += info.get("speed", 0.0)
            total_lane_dev += info.get("lane_dev", 0.0)

            state = next_state
            step += 1

            if reward < -10:
                collided = True

            if render and step % 30 == 0:

                actions = ["LEFT", "RIGHT", "COAST", "ACCEL", "BRAKE"]

                print(
                    f"Step {step:3d} | "
                    f"Speed {info.get('speed',0):5.2f} km/h | "
                    f"LaneDev {info.get('lane_dev',0):.3f} m | "
                    f"Reward {reward:6.2f} | "
                    f"Action {actions[action]}"
                )


        if collided:
            collisions += 1

        avg_speed = total_speed / max(1, step)
        avg_lane_dev = total_lane_dev / max(1, step)

        all_rewards.append(total_reward)
        all_lengths.append(step)
        all_speeds.append(avg_speed)
        all_lane_devs.append(avg_lane_dev)

        print(
            f"[EVAL] Episode {ep:2d} DONE | "
            f"Reward {total_reward:7.2f} | "
            f"Length {step:3d} | "
            f"Speed {avg_speed:5.2f} km/h | "
            f"LaneDev {avg_lane_dev:.3f} m | "
            f"Collided {collided}"
        )

        if render:
            print("Pausing 2s before next episode...")
            time.sleep(2)


    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Episodes:       {EPISODES}")
    print(f"Avg Reward:     {np.mean(all_rewards):.2f}")
    print(f"Avg Length:     {np.mean(all_lengths):.1f} steps")
    print(f"Avg Speed:      {np.mean(all_speeds):.2f} km/h")
    print(f"Avg Lane Dev:   {np.mean(all_lane_devs):.3f} m")
    print(f"Collision Rate: {collisions / EPISODES:.0%}")

    print("=" * 60)

    env.close()


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ddqn_carla_final.pth"

    render = True
    if len(sys.argv) > 2:
        render = sys.argv[2].lower() != "norender"

    evaluate(path, render)