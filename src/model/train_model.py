import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Détection GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations avec augmentation
tf = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Données
train_loader = DataLoader(
    datasets.MNIST("../data/raw", train=True, download=True, transform=tf),
    batch_size=64, shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST("../data/raw", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=False
)

# Modèle CNN amélioré
class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Entraînement
def train(model, n_epoch=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epoch):
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"epoch={epoch}, step={step}: train loss={loss.item():.4f}")

# Test
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"test loss={test_loss:.4f}, accuracy={accuracy:.4f}")

# Main
if __name__ == "__main__":
    input_size = 28 * 28
    output_size = 10
    n_kernels = 32

    model = ConvNet(input_size, n_kernels, output_size).to(device)
    print(f"Parameters={sum(p.numel() for p in model.parameters())/1e3:.3f}K")

    train(model, n_epoch=5)
    test(model)

    os.makedirs("../model", exist_ok=True)
    torch.save(model.state_dict(), "../model/mnist-0.0.1.pt")
