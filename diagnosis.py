# diagnose.py
import numpy as np
import os
import torch

# Check logs
print("=== LOGS ===")
for f in ["logs/rewards.npy", "logs/epsilon.npy", "logs/avg_speed.npy"]:
    if os.path.exists(f):
        data = np.load(f)
        print(f"{f:<30} length: {len(data)}  first_eps: {data[0]:.4f}  last_eps: {data[-1]:.4f}")

# Check checkpoints
print("\n=== CHECKPOINTS ===")
import glob
for path in sorted(glob.glob("checkpoints/*.pth")):
    ck  = torch.load(path, map_location="cpu")
    ep  = ck.get("episode", "?")
    eps = ck.get("epsilon", "?")
    print(f"{os.path.basename(path):<35} episode: {ep}   epsilon: {eps:.4f}")