import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Check if GPU is available and use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic data
# We'll use 100 samples for two classes (0 and 1)
torch.manual_seed(42)  # For reproducibility
n_samples = 100
x0 = torch.randn(n_samples, 2) + torch.tensor([2, 2])  # Class 0
x1 = torch.randn(n_samples, 2) + torch.tensor([-2, -2])  # Class 1

# Labels
y0 = torch.zeros(n_samples)
y1 = torch.ones(n_samples)

# Combine the data
x_data = torch.cat([x0, x1], dim=0)
y_data = torch.cat([y0, y1], dim=0)

# Move data to the device
x_data, y_data = x_data.to(device), y_data.to(device)

# Define a simple feed-forward neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNet().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_data).squeeze()  # Squeeze to remove extra dimensions
    loss = criterion(outputs, y_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store loss
    losses.append(loss.item())
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Evaluate the model on training data
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(x_data).squeeze()
    predicted_labels = (predictions >= 0.5).float()  # Convert probabilities to binary labels
    
    accuracy = (predicted_labels == y_data).float().mean()
    print(f"Training Accuracy: {accuracy.item() * 100:.2f}%")

# Visualize the decision boundary
def plot_decision_boundary(model, x_data, y_data):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1))
    grid = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], dim=2).reshape(-1, 2).to(device)
    
    with torch.no_grad():
        model.eval()
        predictions = model(grid).reshape(xx.shape).cpu()  # Move back to CPU for plotting
    
    plt.contourf(xx, yy, predictions, alpha=0.7, cmap=plt.cm.RdYlBu)
    plt.scatter(x_data[:, 0].cpu(), x_data[:, 1].cpu(), c=y_data.cpu(), s=40, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot the decision boundary
plot_decision_boundary(model, x_data, y_data)
