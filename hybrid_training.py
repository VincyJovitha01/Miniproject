# hybrid_training.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# -------------------------
# Configurations
# -------------------------
n_qubits = 8
chunk_size = n_qubits  # features per re-upload
input_features = 2048  # fingerprint size
encoder_output = 64    # classical encoder output
num_chunks = encoder_output // chunk_size  # 64 → 8 chunks

n_layers_var = 1
epochs = 5
batch_size = 16
lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Quantum device
# -------------------------
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    inputs: 1D tensor of shape (encoder_output,)
    weights: (n_layers_var, n_qubits, 3)
    """
    chunks = inputs.reshape(num_chunks, chunk_size)
    for c in range(num_chunks):
        qml.AngleEmbedding(chunks[c], wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# -------------------------
# Hybrid model
# -------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, encoder_output)
        )
        # Quantum layer
        weight_shapes = {"weights": (n_layers_var, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        # Classical head
        self.classical = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 classes
        )

    def forward(self, x):
        # Encode
        x_enc = self.encoder(x)
        # Quantum layer: process one sample at a time
        quantum_outs = []
        for xi in x_enc:
            quantum_outs.append(self.q_layer(xi))
        x_q = torch.stack(quantum_outs)
        # Classical head
        return self.classical(x_q)

# -------------------------
# Load data
# -------------------------
print("Loading data...")
X = np.load("features.npy")
y = np.load("labels.npy")
print("X shape:", X.shape, "y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# -------------------------
# Model, optimizer, loss
# -------------------------
model = HybridModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Training loop
# -------------------------
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for X_batch, y_batch in pbar:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

print("✅ Training finished.")

# -------------------------
# Evaluation
# -------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    y_proba = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()

y_test_np = y_test_t.cpu().numpy()
acc = accuracy_score(y_test_np, y_pred)
f1 = f1_score(y_test_np, y_pred, average="weighted")
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_np)
try:
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
except:
    roc_auc = float("nan")

print("\n--- Evaluation ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"ROC-AUC: {roc_auc}")
print("------------------")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "hybrid_model_encoder.pth")
print("Model saved to hybrid_model_encoder.pth")
