# Import Libraries
import boto3
import io
import pandas as pd
import random
import os, pickle
import torch, torchtext
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F

# Manual Seed
SEED = 43
torch.manual_seed(SEED)

# Download dataset
s3 = boto3.client('s3', aws_access_key_id= 'AKIAJQ4G7Y5I3HD33SMA',
                  aws_secret_access_key= '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')
obj = s3.get_object(Bucket='tsaibucket', Key='tweets.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

print(df.shape)
print(df.labels.value_counts())

# Defining Fields

Tweet = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
Label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)
fields = [('tweets', Tweet),('labels',Label)]
example = [data.Example.fromlist([df.tweets[i],df.labels[i]], fields) for i in range(df.shape[0])]
twitterDataset = data.Dataset(example, fields)
(train, valid) = twitterDataset.split(split_ratio=[0.85, 0.15], random_state=random.seed(SEED))
print((len(train), len(valid)))

print(vars(train.examples[10]))

Tweet.build_vocab(train)
Label.build_vocab(train)

print('Size of input vocab : ', len(Tweet.vocab))
print('Size of label vocab : ', len(Label.vocab))
print('Top 10 words appreared repeatedly :', list(Tweet.vocab.freqs.most_common(10)))
print('Labels : ', Label.vocab.stoi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iterator, valid_iterator = data.BucketIterator.splits((train, valid), batch_size = 32,
                                                            sort_key = lambda x: len(x.tweets),
                                                            sort_within_batch=True, device = device)

with open('/home/ubuntu/tweetsa/models/tokenizer.pkl', 'wb') as tokens:
    pickle.dump(Tweet.vocab.stoi, tokens)

# define model

class classifier(nn.Module):

    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.encoder = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers,
                               dropout=dropout,
                               batch_first=True)
        # try using nn.GRU or nn.RNN here and compare their performances
        # try bidirectional and compare their performances

        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)

        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # Hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function softmax
        output = F.softmax(dense_outputs[0], dim=1)

        return output

# Define hyperparameters
size_of_vocab = len(Tweet.vocab)
embedding_dim = 300
num_hidden_nodes = 100
num_output_nodes = 3
num_layers = 2
dropout = 0.2

# Instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout = dropout)
print(model)


# No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    _, predictions = torch.max(preds, 1)

    correct = (predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc


# push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

# train loop
def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        tweet, tweet_lengths = batch.tweets

        # convert to 1D tensor
        predictions = model(tweet, tweet_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.labels)

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.labels)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#evaluate loop
def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            tweet, tweet_lengths = batch.tweets

            # convert to 1d tensor
            predictions = model(tweet, tweet_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.labels)
            acc = binary_accuracy(predictions, batch.labels)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '/home/ubuntu/tweetsa/models/saved_weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% \n')
