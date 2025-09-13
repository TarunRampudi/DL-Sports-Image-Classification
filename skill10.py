import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
with open("C:\\Users\\rampu\\OneDrive\\Desktop\\3.2\\Deep learning\\shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

# Encode the entire text
encoded_text = [char2idx[ch] for ch in text]
def create_dataset(sequence, seq_length):
    inputs, targets = [], []
    for i in range(len(sequence) - seq_length):
        inputs.append(sequence[i:i + seq_length])
        targets.append(sequence[i + seq_length])
    return torch.tensor(inputs), torch.tensor(targets)


seq_length = 100
X, y = create_dataset(encoded_text, seq_length)

# Use a smaller subset for fast training (optional)
X = X[:10000]
y = y[:10000]

# Dataset and DataLoader
batch_size = 64
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # only last output
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharRNN(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# ==========================
# Generate Text
# ==========================
def generate_text(model, start_text, length=200, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([char2idx[c] for c in start_text.lower()]).unsqueeze(0).to(device)

    generated_text = start_text

    for _ in range(length):
        # Get the model's prediction
        out = model(input_seq[:, -seq_length:])

        # Apply temperature to the logits (output of the model)
        logits = out / temperature
        probabilities = torch.softmax(logits, dim=-1)

        # Sample a character based on the probability distribution
        pred_idx = torch.multinomial(probabilities, 1).item()

        # Append the predicted character to the generated text
        generated_text += idx2char[pred_idx]

        # Update the input sequence for the next prediction
        input_seq = torch.cat([input_seq, torch.tensor([[pred_idx]]).to(device)], dim=1)

        # Debugging: print the current generated sequence
        print(f"Generated so far: {generated_text[-100:]}")  # Show the last 100 characters generated

    return generated_text


# Example: Generate text
prompt = "to be, or not to be"
generated = generate_text(model, start_text=prompt, length=50, temperature=1.0)

# Print the final generated text
print("\nGenerated Text:\n" + "-" * 20)
print(generated)
