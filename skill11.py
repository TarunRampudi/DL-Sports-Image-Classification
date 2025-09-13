import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================
# Dummy Dataset (Replace with Your Data)
# ============================
# Example: 1000 samples, 20 time steps, 10 features
num_samples = 1000
sequence_length = 20
input_size = 10
num_classes = 2  # Binary classification

# Random input data and labels (replace with your actual dataset)
X = torch.randn(num_samples, sequence_length, input_size)
y = torch.randint(0, num_classes, (num_samples,))

# Dataset and DataLoader
batch_size = 64
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ============================
# LSTM Model Definition
# ============================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.fc(out)
        return out

# ============================
# Model Setup
# ============================
hidden_size = 128
num_layers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_size, hidden_size, num_classes, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# Training Loop
# ============================
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ============================
# Prediction Example
# ============================
model.eval()
sample = X[0].unsqueeze(0).to(device)  # Shape: (1, seq_len, input_size)
output = model(sample)
predicted_class = torch.argmax(output, dim=1).item()
print("\nPredicted class for sample 0:", predicted_class)
