import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_losses(loss_csv_path, title=None):
    loss_data = pd.read_csv(loss_csv_path)

    # Extract relevant data for plotting
    epochs = loss_data['epoch']
    train_loss = loss_data['average_train_loss']
    val_loss = loss_data['average_val_loss']

    #take min value of val loss
    y_min_val_loss = val_loss.min()
    x_min_val_loss = val_loss.idxmin() + 1


    # Plot the training and val losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[0:], val_loss[0:], label='val loss', marker='', linestyle='-', color='blue')  # val Loss line with markers
    plt.plot(epochs[0:], train_loss[0:], label='train loss', marker='', linestyle='-', color='darkorange')  # Training Loss line without markers
    plt.axvline(x=x_min_val_loss, color='grey', linestyle='--')  # Min val Loss vertical line
    plt.plot(x_min_val_loss, y_min_val_loss, 'ro', label='min val loss')  # Min val Loss point
    if title:
        plt.title(title)
    else:
        plt.title('Training vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.splitext(loss_csv_path)[0] + '.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    plot_losses(r"E:\trained_models\pose_RQ1\loss_PoseHRNet-W32_288x384.csv",
                title="HRNet-W32_288x384")

