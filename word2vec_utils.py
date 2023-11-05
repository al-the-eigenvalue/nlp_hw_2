from tqdm import tqdm

import torch
from torch.utils.data import Dataset
    

class SkipgramDataset(Dataset):

    def __init__(self,
                 corpus,
                 word2index,
                 window=2,
                 unk_token='UNK',
                 collect_verbose=True):

        self.corpus = corpus
        self.word2index = word2index
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.window = window

        self.unk_token = unk_token
        self.unk_index = self.word2index[self.unk_token]

        self.collect_verbose = collect_verbose

        self.data = []

        self.collect_data()

    def __len__(self):

        return len(self.data)

    def _split_function(self, tokenized_text):
    
        splits = []
        
        for i in range(len(tokenized_text)):
            for j in range(i-self.window, i+self.window+1):
                if i != j and j >= 0 and j < len(tokenized_text):
                    splits.append((tokenized_text[j], tokenized_text[i]))

        return splits

    def indexing(self, tokenized_text):

        return [self.word2index[token] if token in self.word2index else self.unk_index for token in tokenized_text]

    def collect_data(self):

        corpus = tqdm(self.corpus, disable=not self.collect_verbose)

        for tokenized_text in corpus:
            indexed_text = self.indexing(tokenized_text)
            skipgram_examples = self._split_function(indexed_text)

            self.data.extend(skipgram_examples)

    def __getitem__(self, idx):
        
        context, central_word = self.data[idx]

        return context, central_word
    

class SkipGram(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
    
        super().__init__()
        
        self.in_embedding = torch.nn.Embedding(embedding_dim=embedding_dim,
                                                num_embeddings=vocab_size)
        
        self.out_embedding = torch.nn.Linear(in_features=embedding_dim,
                                                out_features=vocab_size, bias=False)
        
    def forward(self, x):
        
        x = self.in_embedding(x)
        x = self.out_embedding(x)
        
        return x