import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from google.colab import drive

# Mount your Google Drive
drive.mount('/content/drive')

# -----------------------
# Data Utilities
# -----------------------
def one_hot_encode(seq, max_len=23):
    """One-hot encode DNA sequence without flattening (for CNN)."""
    mapping = {"A":0, "C":1, "G":2, "T":3, "N":4}
    encoding = torch.zeros((max_len, len(mapping)))  # [seq_len, channels]
    for i, base in enumerate(seq):
        if i >= max_len:
            break
        idx = mapping.get(base.upper(), 4)
        encoding[i, idx] = 1.0
    return encoding  # keep shape [seq_len, channels]

class gRNADataset(Dataset):
    def __init__(self, pairs, labels, max_len=23):
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq1, seq2 = self.pairs[idx]
        x1 = one_hot_encode(seq1, self.max_len).permute(1,0)  # [channels, seq_len]
        x2 = one_hot_encode(seq2, self.max_len).permute(1,0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, label

# -----------------------
# CNN Encoder
# -----------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# -----------------------
# Siamese Network
# -----------------------
class SiameseNetwork(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = CNNEncoder(in_channels, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        combined = torch.cat([torch.abs(e1 - e2), e1 * e2], dim=-1)
        return self.fc_out(combined).squeeze(-1)

# -----------------------
# Training + Evaluation
# -----------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            outputs = model(x1, x2)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1

def load_pairs_and_labels(csv_file):
    sample_pairs, sample_labels = [], []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            on_seq = row["on_seq"].replace("-", "").replace("_", "")
            off_seq = row["off_seq"].replace("-", "").replace("_", "")
            sample_pairs.append((on_seq, off_seq))
            sample_labels.append(float(row["label"]))
    return sample_pairs, sample_labels

# -----------------------
# Main
# -----------------------
def main():
    csv_file = "/content/drive/MyDrive/Bioinformatics algorithm/CIRCLE_seq.csv"
    sample_pairs, sample_labels = load_pairs_and_labels(csv_file)

    # Train/test split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        sample_pairs, sample_labels, test_size=0.2, random_state=42, stratify=sample_labels
    )

    train_dataset = gRNADataset(train_pairs, train_labels)
    test_dataset = gRNADataset(test_pairs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(5):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, f1 = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()