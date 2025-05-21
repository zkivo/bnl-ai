import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv(r'marco\datasets\Top1k\annotations.csv')

# Split into train, val, test
df_train = df.iloc[:800]
df_val = df.iloc[800:900]
df_test = df.iloc[900:1000]

# Get keypoint columns (exclude filename and bbox)
keypoint_cols = [col for col in df.columns if '-x' in col or '-y' in col]
keypoint_names = sorted(set(col.rsplit('-', 1)[0] for col in keypoint_cols))

def compute_missing_stats(df_split):
    stats = {}
    for kp in keypoint_names:
        if 'bbox' in kp:
            continue
        x_col = f"{kp}-x"
        y_col = f"{kp}-y"
        missing = df_split[[x_col, y_col]].isnull().any(axis=1).sum()
        visible = len(df_split) - missing
        stats[kp] = {"Visible": visible, "Missing": missing}
    return pd.DataFrame(stats).T

# Compute stats
train_stats = compute_missing_stats(df_train)
val_stats = compute_missing_stats(df_val)
test_stats = compute_missing_stats(df_test)

# Combine into one table
summary_stats = pd.concat([
    train_stats.rename(columns={"Visible": "Train Visible", "Missing": "Train Missing"}),
    val_stats.rename(columns={"Visible": "Val Visible", "Missing": "Val Missing"}),
    test_stats.rename(columns={"Visible": "Test Visible", "Missing": "Test Missing"})
], axis=1)

# Display stats table
print(summary_stats)

# Optional: Heatmap of missingness per keypoint per split
summary_stats_pct = summary_stats.copy()
for col in summary_stats.columns:
    if "Visible" in col or "Missing" in col:
        total = 800 if "Train" in col else 100
        summary_stats_pct[col] = (summary_stats[col] / total) * 100

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(summary_stats_pct, annot=True, fmt=".1f", cmap="viridis")
plt.title("Percentage of Visible / Missing Keypoints by Split")
plt.ylabel("Keypoint")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
