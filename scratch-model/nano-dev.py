# This file contains basic component to train a language model from scratch

import torch 
import toch.nn as nn
from toch.nn import function as F


# Placeholder for downloading a dataset 
# 


# Read data set and inspecet it
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# Create uniuqe set for vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from chars to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l)
# Todo, use tiktoken lib for different encoding


# encode the dataset and store in torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into training and validation set
n = int(0.9*len(data))        # 90% of data is training
train_data = data[:n]
val_data = data[n:]

# Data loader
torch.manual_seed(1337)
batch_size = 4
block_size = 8            # Context length

def get_batch(split):
  # Generate a small batch of data of inputs x and target y
  data = train_data if split =='train' else val_data
  ix = torch.randint(len(data) - block_size (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix)
  y = torch.stack([data[i+1:i+block_size+1] for i in ix)
  return x,y


xb, yb = get_batch('train')

# BigramLanguage model 

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


  def forward(self, idx, targets):
    logits = self.token_embedding_table(idx)
    return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, by)


# Pytorch Optimier 
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(100): 

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()






