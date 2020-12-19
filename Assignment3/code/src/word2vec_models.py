import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 200)
        self.fc2 = nn.Linear(200, vocab_size)

    def forward(self, x):
        embed = self.embeddings(x)
        embed = embed.view(embed.shape[0],-1)

        out = F.relu(self.fc1(embed))
        out = self.fc2(out)
        return out

    def ix_to_embeddding(self, ix):
        return self.embeddings(ix)

class Skipgram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Skipgram, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, context_size * vocab_size)

    def forward(self, x):
        embed = self.embeddings(x)
        #embed = embed.view(embed.shape[0], -1)

        out = F.relu(self.fc1(embed))
        out = self.fc2(out)
        out = out.view(out.shape[0], self.context_size, -1)
        return out

    def ix_to_embedding(self, ix):
        return self.embeddings(ix)

