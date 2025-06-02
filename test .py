import torch
from torch import nn

# Define a simple model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the selected device
net = Net().to(device)
print(f"Model is on: {next(net.parameters()).device}")

# Create a sample tensor and move it to the selected device
inputs = torch.randn(1, 10).to(device)
print(f"Inputs are on: {inputs.device}")
