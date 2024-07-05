import torch
import torch.nn as nn
import torch.optim as optim


class BinaryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb):
        super(BinaryPredictor, self).__init__()
        self.nb = nb
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, nb)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def generate_binary_representation(n, nb):
    return [int(b) for b in format(n, f"0{nb}b")]


def loss_fn(outputs, targets):
    return nn.BCELoss()(outputs, targets)


# Example usage
input_dim = 10  # Dimension of the input features
hidden_dim = 64  # Dimension of the hidden layers
nb = 8  # Maximum number of bits to predict

model = BinaryPredictor(input_dim, hidden_dim, nb)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate some example data
batch_size = 16
inputs = torch.randn(batch_size, input_dim)
integer_targets = torch.randint(0, 2**nb, (batch_size,))
binary_targets = torch.tensor(
    [generate_binary_representation(t.item(), nb) for t in integer_targets],
    dtype=torch.float32,
)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, binary_targets)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Inference example
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    output = model(test_input)
    predicted_bits = (output > 0.5).int()
    predicted_integer = int("".join(map(str, predicted_bits.numpy()[0])), 2)
    print(f"Predicted integer: {predicted_integer}")
