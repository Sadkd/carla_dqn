import matplotlib.pyplot as plt

def load_rewards(log_file="logs/rewards.txt"):
    rewards = []
    with open(log_file, "r") as f:
        for line in f:
            rewards.append(float(line.strip()))
    return rewards

def moving_average(data, window=50):
    return [sum(data[max(0,i-window):i+1])/(i-max(0,i-window)+1) for i in range(len(data))]

if __name__ == "__main__":
    rewards = load_rewards()
    smooth = moving_average(rewards)

    plt.figure()
    plt.plot(rewards, label="Raw Reward")
    plt.plot(smooth, label="Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.show()
