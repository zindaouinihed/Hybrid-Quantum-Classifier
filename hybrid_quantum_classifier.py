import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import matplotlib.pyplot as plt

# ======================
# 1. DATA
# ======================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# cats = 3 / dogs = 5
data = [
    (img, 0 if label == 3 else 1)
    for img, label in dataset
    if label in [3, 5]
]

data = data[:800]

loader = DataLoader(
    data,
    batch_size=16,
    shuffle=True
)

# ======================
# 2. CNN
# ======================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 16, 3),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(32 * 14 * 14, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

cnn = CNN()

# ======================
# 3. QUANTUM CIRCUIT
# ======================

x1 = Parameter('x1')
x2 = Parameter('x2')

qc = QuantumCircuit(2)

# Superposition
qc.h(0)
qc.h(1)

# Encoding
qc.ry(x1, 0)
qc.ry(x2, 1)

# Entanglement
qc.cx(0, 1)
qc.cz(0, 1)

# Second quantum layer
qc.ry(x1, 0)
qc.ry(x2, 1)

# Stronger entanglement
qc.cx(1, 0)

# ======================
# SAVE QUANTUM CIRCUIT GRAPH
# ======================

fig = qc.draw(output='mpl')

fig.savefig("quantum_circuit.png")

print("Quantum circuit graph saved!")

# ======================
# 4. QNN
# ======================

observable = Pauli("ZZ")

qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=[x1, x2]
)

model_qnn = TorchConnector(qnn)

# ======================
# 5. HYBRID MODEL
# ======================

class HybridModel(nn.Module):

    def __init__(self, cnn, qnn):
        super().__init__()

        self.cnn = cnn
        self.qnn = qnn

        self.fc = nn.Linear(1, 1)

    def forward(self, x):

        x = self.cnn(x)

        x = self.qnn(x)

        x = self.fc(x)

        return x

model = HybridModel(cnn, model_qnn)

# ======================
# 6. TRAINING
# ======================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

loss_fn = nn.BCEWithLogitsLoss()

losses = []

epochs = 30

for epoch in range(epochs):

    total_loss = 0

    for imgs, labels in loader:

        imgs = imgs.float()

        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs} - Loss = {avg_loss:.4f}")

# ======================
# 7. TRAINING LOSS GRAPH
# ======================

plt.figure(figsize=(8, 5))

plt.plot(losses, marker='o')

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid()

plt.savefig("training_loss.png")

plt.show()

print("Training graph saved!")

print("Training graph saved!")

print("Training completed successfully!")



        
  

