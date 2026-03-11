# ─────────────────────────────────────────────────────────────
# IMDB Sentiment Analysis with Bidirectional LSTM
# Accuracy: 86.08% | Author: Rohith Baskaran
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset
import re
import os

# ── 1. Data ───────────────────────────────────────────────────
raw_dataset = load_dataset('imdb')
train_data  = raw_dataset['train']
test_data   = raw_dataset['test']

print(f"Train: {len(train_data)} reviews | Test: {len(test_data)} reviews")

# ── 2. Tokenizer + Vocabulary ─────────────────────────────────
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r'<.*?>',       ' ',    text)  # remove HTML tags
    text = re.sub(r'([!?.,])',    r' \1 ', text)  # keep punctuation as tokens
    text = re.sub(r'[^a-z0-9\s!?.,]', '', text)  # remove other symbols
    return text.split()

def build_vocab(dataset, max_vocab=25000):
    counter = Counter()
    for item in dataset:
        counter.update(simple_tokenizer(item['text']))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, count in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab

print("Building vocabulary...")
vocab = build_vocab(train_data)
print(f"Vocabulary size: {len(vocab)}")

def encode_text(text, vocab, max_len=200):
    tokens = simple_tokenizer(text)[:max_len]
    return [vocab.get(token, 1) for token in tokens]

# ── 3. Dataset ────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_len=200):
        self.data = []
        for item in hf_dataset:
            encoded = encode_text(item['text'], vocab, max_len)
            label   = item['label']
            self.data.append((encoded, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts         = [torch.tensor(t, dtype=torch.long) for t in texts]
    labels        = torch.tensor(labels, dtype=torch.float)
    texts_padded  = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, labels

# ── 4. DataLoaders ────────────────────────────────────────────
print("Creating datasets... (takes 1-2 mins)")
train_ds = IMDBDataset(train_data, vocab)
test_ds  = IMDBDataset(test_data,  vocab)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_dl)} | Test batches: {len(test_dl)}")

# ── 5. Model ──────────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim,
                                 hidden_dim,
                                 num_layers    = n_layers,
                                 batch_first   = True,
                                 dropout       = 0.3,
                                 bidirectional = True)
        self.fc      = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded                = self.dropout(self.embedding(x))
        output, (hidden, cell)  = self.lstm(embedded)
        hidden                  = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden                  = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)

# ── 6. Training Setup ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model     = SentimentLSTM(
                vocab_size = len(vocab),
                embed_dim  = 128,
                hidden_dim = 256,
                n_layers   = 2
            ).to(device)

Loss      = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── 7. Training Loop ──────────────────────────────────────────
losses  = []
epoches = []
n       = len(train_dl)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, (text, label) in enumerate(train_dl):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss   = Loss(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())
        epoches.append(epoch + i / n)
        if i % 100 == 99:
            print('[%d, %3d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
            running_loss = 0.0

print('Finished Training')

# ── 8. Save Model ─────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab':            vocab,
    'model_config': {
        'vocab_size': len(vocab),
        'embed_dim':  128,
        'hidden_dim': 256,
        'n_layers':   2
    }
}, 'sentiment_lstm.pth')
print("Model saved!")

# ── 9. Evaluation ─────────────────────────────────────────────
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for text, label in test_dl:
        text, label = text.to(device), label.to(device)
        output      = model(text)
        predicted   = (torch.sigmoid(output) > 0.5).float()
        total      += label.size(0)
        correct    += (predicted == label).sum().item()

print(f'\nTest Accuracy: {100 * correct / total:.2f}%')

# ── 10. Sample Predictions ────────────────────────────────────
def predict_sentiment(review, model, vocab, device):
    model.eval()
    with torch.no_grad():
        encoded   = encode_text(review, vocab, max_len=200)
        tensor    = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        output    = model(tensor)
        prob      = torch.sigmoid(output).item()
        sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
        confidence = prob if prob > 0.5 else 1 - prob
    print(f"Review:     {review[:60]}...")
    print(f"Sentiment:  {sentiment}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("-" * 50)

reviews = [
    "This movie was absolutely brilliant, loved every second of it",
    "Worst film I have ever seen, complete waste of time",
    "It was okay, nothing special but not terrible either",
    "The acting was poor but the story was surprisingly good",
    "I fell asleep halfway through, extremely boring and slow",
]

print("\nSample Predictions:")
for review in reviews:
    predict_sentiment(review, model, vocab, device)

# ── 11. Loss Curve ────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

epoch_avg = np.array(epoches).reshape(10, -1).mean(axis=1)
loss_avg  = np.array(losses).reshape(10, -1).mean(axis=1)

plt.figure(figsize=(10, 4))
plt.plot(epoch_avg, loss_avg, color='violet', linewidth=2, marker='o', markersize=5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Sentiment Analysis — Training Loss')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('results/loss_curve.png')
plt.show()
print("Loss curve saved!")