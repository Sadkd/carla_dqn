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

def set_spectator(world, vehicle):
    """Move the CARLA spectator camera to follow the vehicle from behind."""
    transform = vehicle.get_transform()

    # Position camera behind and above the vehicle
    spectator_transform = carla.Transform(
        carla.Location(
            x=transform.location.x - 6.0 * transform.get_forward_vector().x,
            y=transform.location.y - 6.0 * transform.get_forward_vector().y,
            z=transform.location.z + 3.5
        ),
        carla.Rotation(
            pitch=-15.0,          # tilt down slightly to see the road
            yaw=transform.rotation.yaw,
            roll=0.0
        )
    )
    world.get_spectator().set_transform(spectator_transform)

def evaluate(checkpoint_path="checkpoints/ddqn_carla_final.pth", render=True):
    env = CarlaEnv(stack_size=FRAME_STACK)
    agent = DQNAgent(
        device='cpu',
        frame_stack=FRAME_STACK,
        num_actions=NUM_ACTIONS
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_model_state_dict"])
    agent.policy_net.eval()
    agent.epsilon = 0.01  # almost no exploration

    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {checkpoint.get('episode', '?')} episodes")
    print(f"  Saved epsilon: {checkpoint.get('epsilon', '?'):.4f}")
    print(f"  Render mode: {'ON — watch in CARLA window' if render else 'OFF'}")
    print("=" * 50)

    # Get CARLA world reference for spectator
    world = env.world if render else None

    all_rewards   = []
    all_lengths   = []
    all_speeds    = []
    all_lane_devs = []
    collisions    = 0

    for ep in range(EPISODES):
        state = env.reset()                        # shape: (stack_size, H, W, 3)
        stacked_state = state.reshape(-1, 84, 84)  # → (stack_size*3, H, W)

        done         = False
        total_reward = 0.0
        total_speed  = 0.0
        total_lane   = 0.0
        step         = 0
        collided     = False

        print(f"\n[EVAL] Episode {ep} starting...")

        while not done:
            # --- Update spectator camera to follow vehicle ---
            if render and env.vehicle is not None:
                set_spectator(world, env.vehicle)

            # --- Select and perform action ---
            action = agent.select_action(stacked_state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_speed  += info.get("speed", 0.0)
            total_lane   += info.get("lane_dev", 0.0)

            stacked_state = next_state.reshape(-1, 84, 84)
            step += 1

            if reward < -10:
                collided = True

            # --- Live step info ---
            if render and step % 30 == 0:  # print every 30 steps
                print(
                    f"  Step {step:3d} | "
                    f"Speed: {info.get('speed', 0):5.2f} km/h | "
                    f"LaneDev: {info.get('lane_dev', 0):.3f}m | "
                    f"Reward: {reward:6.2f} | "
                    f"Action: {['LEFT','RIGHT','COAST','ACCEL','BRAKE'][action]}"
                )

        if collided:
            collisions += 1

        avg_speed    = total_speed / max(1, step)
        avg_lane_dev = total_lane  / max(1, step)

        all_rewards.append(total_reward)
        all_lengths.append(step)
        all_speeds.append(avg_speed)
        all_lane_devs.append(avg_lane_dev)

        print(
            f"[EVAL] Episode {ep:2d} DONE | "
            f"Reward: {total_reward:7.2f} | "
            f"Length: {step:3d} | "
            f"Speed: {avg_speed:5.2f} km/h | "
            f"LaneDev: {avg_lane_dev:.3f}m | "
            f"Collided: {collided}"
        )

        # Pause between episodes so you can see the reset
        if render:
            print("  Pausing 2s before next episode...")
            time.sleep(2.0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:        {EPISODES}")
    print(f"Avg Reward:      {np.mean(all_rewards):.2f}")
    print(f"Avg Length:      {np.mean(all_lengths):.1f} steps")
    print(f"Avg Speed:       {np.mean(all_speeds):.2f} km/h")
    print(f"Avg Lane Dev:    {np.mean(all_lane_devs):.3f} m")
    print(f"Collision Rate:  {collisions / EPISODES:.0%}")
    print("=" * 50)

    env.close()

if __name__ == "__main__":
    # Usage:
    #   python evaluate.py                                          → final checkpoint, render ON
    #   python evaluate.py checkpoints/ddqn_carla_ep1200.pth       → specific checkpoint, render ON
    #   python evaluate.py checkpoints/ddqn_carla_ep1200.pth norender → render OFF

    path   = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ddqn_carla_final.pth"
    render = sys.argv[2].lower() != "norender" if len(sys.argv) > 2 else True

    evaluate(path, render)