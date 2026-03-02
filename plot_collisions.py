import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window=50):
    return [
        np.mean(data[max(0, i-window):i+1])
        for i in range(len(data))
    ]

if __name__ == "__main__":

    collisions = np.load("logs/collision_rate.npy")

    rate = moving_average(collisions, window=50)

    plt.figure()
    plt.plot(rate)
    plt.xlabel("Episode")
    plt.ylabel("Collision Rate (Moving Avg)")
    plt.title("Collision Rate Over Time")
    plt.grid()
    plt.show()
