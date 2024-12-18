import matplotlib.pyplot as plt

# Training loss (every 3rd epoch)
epochs = list(range(3, 58, 3))  # Every 3rd epoch up to 57
training_loss = [
    0.027144, 0.027010, 0.026877, 0.026809, 0.026800, 
    0.026690, 0.026544, 0.026028, 0.025290, 0.024665, 
    0.024110, 0.023721, 0.023300, 0.023147, 0.023034, 
    0.022928, 0.022832, 0.022606, 0.022399
]

# Evaluation metrics
accuracy = [70.09, 70.00, 70.00, 70.19, 70.48, 70.52, 70.55, 70.61, 70.67, 70.61, 70.63, 70.72, 70.73, 70.72, 70.76, 70.77, 70.76, 70.72, 70.76]
precision = [0.7036, 0.7009, 0.7008, 0.6990, 0.7012, 0.7019, 0.7031, 0.7040, 0.7050, 0.7042, 0.7057, 0.7082, 0.7066, 0.7068, 0.7073, 0.7072, 0.7072, 0.7068, 0.7072]
recall = [0.6944, 0.6978, 0.6981, 0.7094, 0.7136, 0.7133, 0.7114, 0.7112, 0.7110, 0.7105, 0.7078, 0.7048, 0.7091, 0.7082, 0.7083, 0.7091, 0.7085, 0.7082, 0.7091]
f1_score = [0.6990, 0.6993, 0.6994, 0.7042, 0.7073, 0.7076, 0.7072, 0.7075, 0.7080, 0.7074, 0.7067, 0.7065, 0.7079, 0.7075, 0.7078, 0.7074, 0.7078, 0.7080, 0.7081]

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Training Loss and Evaluation Metrics (Every 3rd Epoch)", fontsize=16)

# Plot Training Loss
axes[0, 0].plot(epochs, training_loss, marker='o', linestyle='-', color='blue')
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid()

# Plot Accuracy
axes[0, 1].plot(epochs, accuracy, marker='o', linestyle='-', color='green')
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].grid()

# Plot Precision
axes[0, 2].plot(epochs, precision, marker='o', linestyle='-', color='orange')
axes[0, 2].set_title("Precision")
axes[0, 2].set_xlabel("Epochs")
axes[0, 2].set_ylabel("Precision")
axes[0, 2].grid()

# Plot Recall
axes[1, 0].plot(epochs, recall, marker='o', linestyle='-', color='red')
axes[1, 0].set_title("Recall")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("Recall")
axes[1, 0].grid()

# Plot F1 Score
axes[1, 1].plot(epochs, f1_score, marker='o', linestyle='-', color='purple')
axes[1, 1].set_title("F1 Score")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].grid()

# Hide the last unused subplot
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title
plt.show()
