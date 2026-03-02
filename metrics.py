class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.collisions = 0
        self.episodes = 0

    def log_episode(self, reward, collided):
        self.episode_rewards.append(reward)
        self.episodes += 1
        if collided:
            self.collisions += 1

    def summary(self):
        if self.episodes == 0:
            return {}

        return {
            "avg_reward": sum(self.episode_rewards) / self.episodes,
            "collision_rate": self.collisions / self.episodes,
            "episodes": self.episodes
        }
