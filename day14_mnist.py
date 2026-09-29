import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root = "./data",  train = True, download = True, transform = transform)
test_dataset = datasets.MNIST(root = "./data",  train = False, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)
print(f"训练集：{len(train_dataset)} 张 | 测试集：{len(test_dataset)} 张")

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

model = MNISTModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 5
train_losses, test_accuracies = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            y_pred = model(images)
            predicted = y_pred.argmax(dim = 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {accuracy*100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker="o", color="blue")
plt.title("训练 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in test_accuracies], marker="o", color="green")
plt.title("测试准确率曲线")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("day14_mnist.png")
plt.show()

model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(images)
    predicted = y_pred.argmax(dim=1)

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"真实: {labels[i].item()} | 预测: {predicted[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("day14_predictions.png")
plt.show()
