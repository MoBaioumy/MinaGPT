import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming 'input.txt' is already present and contains your dataset

# Read dataset and inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create unique set for vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from chars to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the dataset and store in torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into training and validation set
n = int(0.9 * len(data))  # 90% of data is for training
train_data = data[:n]
val_data = data[n:]

# Data loader
torch.manual_seed(1337)
batch_size = 4
block_size = 8  # Context length

def get_batch(split):
    # Generate a small batch of data of inputs x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

# BigramLanguage model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.token_embedding_table(idx)
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb)

# PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for steps in range(100):
    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass
    logits = m(xb)

    # compute the loss
    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

def generate_text(model, seed_text, next_words=100):
    model.eval()
    for _ in range(next_words):
        encoded_seed = torch.tensor(encode(seed_text[-block_size:]), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = model(encoded_seed)
        last_word_logits = logits[0, -1]
        predicted_word = torch.multinomial(F.softmax(last_word_logits, dim=0), 1)
        seed_text += decode([predicted_word.item()])
    return seed_text
