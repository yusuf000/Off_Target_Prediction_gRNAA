import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# -----------------------
# Data Utilities
# -----------------------
def one_hot_encode(seq, max_len=24):
    mapping = {"A":0, "C":1, "G":2, "T":3, "N":4}
    encoding = torch.zeros((max_len, len(mapping)))
    seq = seq.replace("_", "")
    for i, base in enumerate(seq):
        if i >= max_len:
            break
        idx = mapping.get(base.upper(), 4)
        encoding[i, idx] = 1.0
    return encoding

class gRNADataset(Dataset):
    def __init__(self, pairs, labels, max_len=24):
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq1, seq2 = self.pairs[idx]
        x1 = one_hot_encode(seq1, self.max_len).permute(1,0)
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
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=1),
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
            nn.Linear(64,1)  # remove Sigmoid
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
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            logits = model(x1, x2)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = 1 / (1 + np.exp(-np.array(all_logits)))  # Sigmoid
    preds = (all_probs > threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    return acc, f1, precision, recall, all_probs

def find_best_threshold(all_probs, all_labels):
    thresholds = np.linspace(0.5, 0.99, 50)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

# -----------------------
# Load Data
# -----------------------
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
    csv_file = "E:\\Bioinformatics\\OffTopicDetection\\siamese-network-grna-off-target\\datasets\\CIRCLE_seq.csv"
    model_path = "E:\\Bioinformatics\\OffTopicDetection\\siamese-network-grna-off-target\\siamese_model\\siamese_model.pth"
    optimizer_path = "E:\\Bioinformatics\\OffTopicDetection\\siamese-network-grna-off-target\\siamese_optimizer\\siamese_optimizer.pth"

    sample_pairs, sample_labels = load_pairs_and_labels(csv_file)

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        sample_pairs, sample_labels, test_size=0.2, random_state=42, stratify=sample_labels
    )

    labels_np = np.array(train_labels)
    num_pos = np.sum(labels_np == 1)
    num_neg = np.sum(labels_np == 0)
    pos_weight = num_neg / num_pos
    print("Positive class count:", num_pos)
    print("Negative class count:", num_neg)
    print("Positive class weight (for BCEWithLogitsLoss):", pos_weight)

    train_dataset = gRNADataset(train_pairs, train_labels)
    test_dataset = gRNADataset(test_pairs, test_labels)

    # Balanced sampler
    class_counts = np.bincount(np.array(train_labels).astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[np.array(train_labels).astype(int)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -----------------------
    # Resume training if checkpoint exists
    # -----------------------
    start_epoch = 0
    if os.path.exists(model_path):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path))
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        start_epoch = 0  # change if you also want to save epoch count

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(start_epoch, 40):  # adjust max epochs
        loss = train(model, train_loader, criterion, optimizer, device)
        acc, f1, precision, recall, all_probs = evaluate(model, test_loader, device)
        best_thresh = find_best_threshold(all_probs, np.array(test_labels))

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, "
              f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"Best Threshold: {best_thresh:.2f}")

        # -----------------------
        # Save checkpoint
        # -----------------------
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"✅ Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    main()