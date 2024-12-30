import matplotlib.pyplot as plt
import pandas as pd

loss_data = pd.read_csv('out/train-241229_132945/loss.csv')

# Extract relevant data for plotting
epochs = loss_data['epoch']
train_loss = loss_data['average_train_loss']
test_loss = loss_data['average_test_loss']

# Plot the training and test losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', marker='', linestyle='-', color='darkorange')  # Training Loss line without markers
plt.plot(epochs, test_loss, label='Test Loss', marker='', linestyle='-', color='orange')  # Test Loss line with markers
plt.title('Training vs. Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
