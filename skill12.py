import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# =====================================
# Dataset and Transformations
# =====================================
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_set, val_set = random_split(dataset, [45000, 5000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# =====================================
# Autoencoder
# =====================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x4x4
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 32x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 3x32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# =====================================
# ResNet Classifier (Simple Variant)
# =====================================
from torchvision.models import resnet18

class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)  # expects input channels=64
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        return self.resnet(x)

# =====================================
# Train Autoencoder
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder().to(device)
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training Autoencoder...")
for epoch in range(10):
    autoencoder.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        output = autoencoder(images)
        loss = criterion(output, images)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# =====================================
# Train Classifier on Encoded Features
# =====================================
classifier = ResNetClassifier().to(device)
clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
clf_criterion = nn.CrossEntropyLoss()

print("\nTraining Classifier...")
for epoch in range(10):
    classifier.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            encoded = autoencoder.encoder(images)
        output = classifier(encoded)
        loss = clf_criterion(output, labels)
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
