import numpy as np

class MetricsTracker:
    def __init__(self):
        self.episode_rewards  = []
        self.collision_counts = []
        self.episode_lengths  = []
        self.avg_speeds       = []
        self.avg_lane_devs    = []
        self.avg_losses       = []
        self.mean_max_qs      = []
        self.epsilons         = []
        self.episodes         = 0

    def log_episode(self, reward, collisions, ep_length,
                    avg_speed, avg_lane_dev, avg_loss, mean_max_q, epsilon):
        self.episode_rewards.append(reward)
        self.collision_counts.append(collisions)
        self.episode_lengths.append(ep_length)
        self.avg_speeds.append(avg_speed)
        self.avg_lane_devs.append(avg_lane_dev)
        self.avg_losses.append(avg_loss)
        self.mean_max_qs.append(mean_max_q)
        self.epsilons.append(epsilon)
        self.episodes += 1

    def summary(self, last_n=100):
        if self.episodes == 0:
            return {}

        n = min(last_n, self.episodes)

        return {
            # Overall
            "episodes":              self.episodes,
            "avg_reward":            np.mean(self.episode_rewards),
            "avg_collision_rate":    np.mean(self.collision_counts),
            "avg_episode_length":    np.mean(self.episode_lengths),
            "avg_speed":             np.mean(self.avg_speeds),
            "avg_lane_dev":          np.mean(self.avg_lane_devs),
            "avg_loss":              np.mean(self.avg_losses),
            "avg_max_q":             np.mean(self.mean_max_qs),

            # Last N episodes
            f"avg_reward_last{n}":   np.mean(self.episode_rewards[-n:]),
            f"avg_speed_last{n}":    np.mean(self.avg_speeds[-n:]),
            f"avg_lane_dev_last{n}": np.mean(self.avg_lane_devs[-n:]),
            f"collision_rate_last{n}": np.mean(self.collision_counts[-n:]),
        }

    def print_summary(self, last_n=100):
        s = self.summary(last_n)
        if not s:
            print("No episodes logged yet.")
            return
        n = min(last_n, self.episodes)
        print("=" * 50)
        print(f"METRICS SUMMARY — {self.episodes} episodes")
        print("=" * 50)
        print(f"Avg Reward (all):       {s['avg_reward']:.2f}")
        print(f"Avg Reward (last {n}):  {s[f'avg_reward_last{n}']:.2f}")
        print(f"Avg Episode Length:     {s['avg_episode_length']:.1f} steps")
        print(f"Avg Speed:              {s['avg_speed']:.2f} km/h")
        print(f"Avg Speed (last {n}):   {s[f'avg_speed_last{n}']:.2f} km/h")
        print(f"Avg Lane Dev:           {s['avg_lane_dev']:.3f} m")
        print(f"Avg Lane Dev (last {n}):{s[f'avg_lane_dev_last{n}']:.3f} m")
        print(f"Avg Collisions/ep:      {s['avg_collision_rate']:.2f}")
        print(f"Collision Rate (last {n}): {s[f'collision_rate_last{n}']:.2%}")
        print(f"Avg Loss:               {s['avg_loss']:.4f}")
        print(f"Avg Max Q:              {s['avg_max_q']:.2f}")
        print("=" * 50)