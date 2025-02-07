import torch
import torch.nn as nn
import torch.optim as optim

class CustomModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModule, self).__init__()
        self.W = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(torch.addmm(self.b, x, self.W))
        x = torch.tanh(self.linear(x))
        return x

# Initialize the model
input_size = 12
hidden_size = 256
output_size = 3
model = CustomModule(input_size, hidden_size, output_size)

# Example input tensor
input_tensor = torch.randn(1, input_size)

# Forward pass
output_tensor = model(input_tensor)

print("Input tensor:", input_tensor)
print("Output tensor:", output_tensor)