# Import libraries
import pandas as pd
import torch
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load data
DATA_PATH = "data/ghg_filtered.csv"
df = pd.read_csv(DATA_PATH)

# Tokenize text from Material Description column of df
tokenizer = get_tokenizer("basic_english")
token_list = [tokenizer(row) for row in df["Material Description"]]
vocab = build_vocab_from_iterator(token_list)
idx_list = [[vocab.stoi[token] for token in row] for row in token_list]

# Pad tokens
seq_lens = [len(idx_list[i]) for i in range(len(idx_list))]
max_seq_len = max(seq_lens)
padded_idx_list = [idx_list[i] + [1] * (max_seq_len -  seq_lens[i]) for i in range(len(idx_list))]

# Preprocess labels
labels = df['CAT 11 (USE)']
labels = [row.lower() for row in labels]
for i, row in enumerate(labels):
    if 'excluded' in row:
        labels[i] = 'excluded'
labels_uni = list(set(labels))
label_map = {k : i for i, k in enumerate(labels_uni)}
label_idx = [label_map[row] for row in labels]

# Create Datasets and DataLoaders
input_text = torch.LongTensor(padded_idx_list)
input_labels = torch.LongTensor(label_idx)
train_text, test_text, train_label, test_label = train_test_split(input_text, input_labels, test_size=0.2, random_state=42)
val_text, test_text, val_label, test_label = train_test_split(test_text, test_label, test_size=0.5, random_state=42)
train_dataset = TensorDataset(train_text, train_label)
val_dataset = TensorDataset(val_text, val_label)
test_dataset = TensorDataset(test_text, test_label)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define Model
class TextClassificationCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassificationCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size = 3, padding = 1, stride = 1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.mean(dim = 2)
        x = self.fc(x)
        return x

model = TextClassificationCNN(len(vocab), 100, len(labels_uni))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
model.to(device)

num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    for data, label in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_avg = running_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {loss_avg}")

# Validate model
model.eval()
running_val_loss = 0.0
with torch.no_grad():
    for data, label in val_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        running_val_loss += loss.item()
        val_loss_avg = running_val_loss / len(val_loader)
    print(f"Validation Loss: {val_loss_avg}")

