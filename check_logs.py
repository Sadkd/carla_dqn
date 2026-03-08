# check_logs.py
import numpy as np
import os

for f in ["logs/rewards.npy", "logs/epsilon.npy"]:
    if os.path.exists(f):
        data = np.load(f)
        print(f"{f:<30} episodes completed: {len(data)}   last value: {data[-1]:.4f}")
    else:
        print(f"{f:<30} NOT FOUND")