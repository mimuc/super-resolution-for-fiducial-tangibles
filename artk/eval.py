import pandas as pd
import numpy as np

files = ["log_artk_small.csv", "log_artk_small_bs.csv", "log_artk_large.csv", "log_artk_large_bs.csv"]

for file in files:
    print(file)
    df = pd.read_csv(file)

    print(df.shape)
    stats = df['Code'].value_counts().to_dict()
    print(stats)
    if not -2 in stats:
        stats[-2] = 0
    if not -1 in stats:
        stats[-1] = 0
    if not 1 in stats:
        stats[1] = 0

    print(f"Acc: {stats[1] / df.shape[0]}, Failed: {stats[-2] / df.shape[0]}")

    df = df.loc[df['Rotation'] != -1]
    mae = np.mean(df['Rotation'])
    std = np.std(df['Rotation'])
    print(f"Rot MAE: {mae}, SD: {std}")

