import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_velocity(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)

# Helper function: compute angle of body vector (in degrees)
def compute_body_angle(nose_x, nose_y, tail_x, tail_y):
    dx = nose_x - tail_x
    dy = nose_y - tail_y
    angles = np.degrees(np.arctan2(dy, dx))
    return angles

# Load the CSV file
csv_path = r"C:\Users\marco\Desktop\20250315 - M421604 - Dyskinesia - 20mgLD 8mgB - 3600s - trim -25-27\20250315 - M421604 - Dyskinesia - 20mgLD 8mgB - 3600s - trim -25-27-inference.csv"
df = pd.read_csv(csv_path)

df.head()

# Extract key keypoints
nose_x, nose_y = df['nose-x'], df['nose-y']
tail_x, tail_y = df['tail_base-x'], df['tail_base-y']
back_x, back_y = df['back_midpoint-x'], df['back_midpoint-y']

# Drop NaNs for clean trajectory
mask = ~nose_x.isna() & ~nose_y.isna()
nose_x, nose_y = nose_x[mask].values, nose_y[mask].values

# 1. Trajectory plot (nose)
plt.figure(figsize=(8, 6))
plt.plot(nose_x, nose_y, lw=1, alpha=0.7)
plt.title("Mouse Nose Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.grid(True)

# 2. Velocity over time (nose)
velocity = compute_velocity(nose_x, nose_y)
plt.figure(figsize=(10, 4))
plt.plot(velocity)
plt.title("Mouse Nose Speed Over Time")
plt.xlabel("Frame")
plt.ylabel("Speed (pixels/frame)")
plt.grid(True)

# 4. Body orientation angle (nose to tail)
angle = compute_body_angle(nose_x[:len(tail_x)], nose_y[:len(tail_y)], tail_x[:len(nose_x)], tail_y[:len(nose_y)])
plt.figure(figsize=(10, 4))
plt.plot(angle)
plt.title("Body Orientation Angle Over Time")
plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.grid(True)
plt.show()