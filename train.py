import numpy as np  # type: ignore
import json
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
from snn import FraudSNN
from torch.utils.data import TensorDataset, DataLoader  # type: ignore
from dataset import load_dataset, rate_code
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns # type: ignore
print("Seaborn version:", sns.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("config.json", "r") as file:
    Config = json.load(file)



def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device="cpu"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses = []
    val_losses = []

    print("\nStarting training on device:", device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch)
                val_loss += loss.item() * X_val_batch.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_val_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / len(val_loader.dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {accuracy:.4f}"
        )
    return model, train_losses, val_losses




def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            outputs = model(X_test_batch)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_test_batch.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    return all_preds, all_labels




def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()





def save_model(model, filename="saved_models/snn_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def save_model_scripted(model, filename="saved_models/snn_scripted.pt"):
    scripted_model = torch.jit.script(model)
    scripted_model.save(filename)
    print(f"Scripted model saved to {filename}")






# Load the dataset
X_train, X_test, X_val, y_val, y_train, y_test = load_dataset(
    Config["dataset_path"], Config["dataset_frac"], Config["validation_split"]
)
# Convert each sample into a spike train. The output shape will be [samples, time_steps, features]
X_train_spikes = np.array([rate_code(row) for row in X_train])
X_val_spikes = np.array([rate_code(row) for row in X_val])
X_test_spikes = np.array([rate_code(row) for row in X_test])

# Prepare PyTorch Datasets and DataLoaders
X_train_tensor = torch.tensor(X_train_spikes, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_spikes, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_spikes, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)





model = FraudSNN(input_size=11, hidden_size=64, time_steps=10, beta=0.9, threshold=1.0)




model, train_loss, val_loss = train_model(
    model, train_loader, val_loader, num_epochs=50, lr=1e-3, device=device
)




all_labels,all_preds = evaluate_model(model, test_loader, device=device)





# -------------------------
# Plot Confusion Matrix
# -------------------------
plot_confusion_matrix(all_labels, all_preds)

# -------------------------
# Plot Loss Curves
# -------------------------
plot_loss(train_loss, val_loss)





save_model(model, filename="saved_models/snn_model.pt")

save_model_scripted(model, filename="saved_models/snn_scripted.pt")