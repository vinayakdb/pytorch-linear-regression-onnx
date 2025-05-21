import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 3).astype(np.float32)  # 100 samples, 3 features
y = (X @ np.array([1.5, -2.0, 3.0]) + 0.5).astype(np.float32).reshape(-1, 1)  # Linear relation with noise

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)  # 3 input features, 1 output

    def forward(self, x):
        return self.linear(x)

# Instantiate the model, define the loss function and the optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")