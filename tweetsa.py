import spacy
import torch, torchtext
nlp = spacy.load('en')
from flask import Flask, jsonify, request, redirect, render_template
import os, pickle
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = "secret key"

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
size_of_vocab = 4651
embedding_dim = 300
num_hidden_nodes = 100
num_output_nodes = 3
num_layers = 2
dropout = 0.2

# Instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout = dropout)

# load weights and tokenizer

path = '/home/ubuntu/tweetsa/models/saved_weights.pt'
model.load_state_dict(torch.load(path));
model.eval();
tokenizer_file = open('/home/ubuntu/tweetsa/models/tokenizer.pkl', 'rb')
tokenizer = pickle.load(tokenizer_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inference

def classify_tweet(tweet):

    # tokenize the tweet
    tokenized = [tok.text for tok in nlp.tokenizer(tweet)]
    # convert to integer sequence using predefined tokenizer dictionary
    indexed = [tokenizer[t] for t in tokenized]
    # compute no. of words
    length = [len(indexed)]
    # convert to tensor
    tensor = torch.LongTensor(indexed).to(device)
    # reshape in form of batch, no. of words
    tensor = tensor.unsqueeze(1).T
    # convert to tensor
    length_tensor = torch.LongTensor(length)
    # Get the model prediction
    prediction = model(tensor, length_tensor)

    _, pred = torch.max(prediction, 1)

    return pred.item()

# URL Routes
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        my_prediction = classify_tweet(tweet)
    return render_template('result.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')