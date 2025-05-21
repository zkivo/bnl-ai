import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r"E:\hrnet_w32_384_288-RMSE-confidence.csv")  # Replace with your file path

# Create the figure and axis
fig, ax1 = plt.subplots()

# Plot RMSE on the left y-axis
color_rmse = 'tab:red'
ax1.set_xlabel('Confidence')
ax1.set_ylabel('RMSE', color=color_rmse)
ax1.plot(df['confidence'], df['RMSE'], color=color_rmse, label='RMSE')
ax1.tick_params(axis='y', labelcolor=color_rmse)

# Create a second y-axis for coverage
ax2 = ax1.twinx()
color_coverage = 'tab:blue'
ax2.set_ylabel('Coverage', color=color_coverage)
ax2.plot(df['confidence'], df['coverage'], color=color_coverage, label='Coverage')
ax2.tick_params(axis='y', labelcolor=color_coverage)

# dot and vertical line at 0.75
# ax1.plot(0.75, df[df['confidence'] == 0.75]['RMSE'].values[0], 'o', color=color_rmse)
# ax2.plot(0.75, df[df['confidence'] == 0.75]['coverage'].values[0], 'o', color=color_coverage)
ax1.axvline(x=0.50, color='gray', linestyle='--', label='Confidence = 0.50')
ax1.axvline(x=0.95, color='gray', linestyle='--', label='Confidence = 0.95')

# Add a title and show the plot
plt.title('RMSE and Coverage vs Confidence')
fig.tight_layout()
plt.show()
