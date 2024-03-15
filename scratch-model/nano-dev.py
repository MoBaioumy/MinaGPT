# This file contains basic component to train a language model from scratch

import torch 


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



