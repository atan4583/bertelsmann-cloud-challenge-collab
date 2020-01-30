#!/usr/bin/env python3

from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask_cors import CORS, cross_origin

import pandas as pd
import argparse
import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from string import punctuation

from string import punctuation
from collections import Counter

# from datetime import datetime
import os
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

kfv = {}
vocab_to_int = {}
rnk = {
    'anger': 0,
    'disgust': 1,
    'guilt': 2,
    'fear': 3,
    'shame': 4,
    'joy': 5,
    'sadness': 6
}

network_initialized = False
net = None
execute_on_gpu = False

"""

it works like this:

0. copy snapshot of the trained network to the folder above "../" 
1. install flask and flask_cors (conda install -c anaconda flask flask_cors or use pip / pip3) 
2. set the env variable FLASK_APP to process.py, for example in Windows set FLASK_APP=process.py
3. start flask: flask run (and keep the shell open!!!)
4. call the webserver with a tool hat can create post requests, for example curl. This is a sample request: 
    curl -H "Content-type: application/json" -X POST -d '{"review":"I feel bad that I took two plates"}' http://127.0.0.1:5000
    
    the Python file expects
    - the POST data in JSON
    - an entry for "review"
    
    in case an error happens, and there are a lot of possibilities for that, there is zero error handling
    
call http://localhost:5000/refresh to reload the model with the next classification run    
"""


@app.route("/", methods=['POST', 'GET'])
@cross_origin(origin='*' ,headers=['Content- Type', 'Authorization'])
def classify():

    result = ""
    global network_initialized
    global net
    global execute_on_gpu

    """"
    the first time the method is executed, the network get initialized and stored in a global variable
    as well as the the execute_on_cpu variable
    """
    if not network_initialized:
        print("initializing network...")
        net, execute_on_gpu = init()
        network_initialized = True

    if request.headers['Content-Type'] == 'application/json':
        print("JSON Message: " + json.dumps(request.json))
        review = request.json['review']
        print("review: ", review)
        # todo - check if string is empty or whether is has been in the request at all
        result += predict(net, review,  execute_on_gpu, sequence_length=200)
    else:
        result += "no json body found"

    data = {'classification': result, 
            'review': review}
    js = json.dumps(data)

    resp = Response(js, status=200, mimetype='application/json')

    print(result)

    return resp


@app.route("/refresh", methods=['GET'])
@cross_origin(origin='*' ,headers=['Content- Type', 'Authorization'])
def refresh():
    """
        this doesn't do much. It will just set the global network_initialized value to False, so it will be
        reloaded in the next classification attempt
    """
    global network_initialized
    network_initialized = False

    print("set network_initialized to False...")
    return "done"


def main():
    """
    this method will only be called if the file is run from the command line
    it will be ignored in case in the context of execution through flask
    :return:
    """
    net, execute_on_gpu = init()

    in_arg = get_input_args()
    in_args = in_arg.parse_args()

    review = in_args.review
    print("predicting {}...".format(review))
    predict(net, review,  execute_on_gpu, sequence_length=200)


def init():
    """this is the main method, so initialize a few variables,
        parse command line arguments and then start the main logic of this program
    """

    print('pytorch ver:', torch.__version__)
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    execute_on_gpu = torch.cuda.is_available()

    # change chkptdir value to suit
    chkptdir = '../'
    if len(glob.glob(chkptdir + '/*.pt')) > 0:
        # checkpt = 'rnn_classifier.pt'
        checkpt = max(glob.glob(chkptdir + '/*.pt'), key=os.path.getctime)
        print('checkpoint:', checkpt, ' located')
    else:
        checkpt = None
        print('\n*** no saved checkpoint found !!!\n')
        exit(0)

    # Change fpath value to suit
    fpath = '../'
    lblpth = os.path.join(fpath, 'labels.txt')
    sentimentpth = os.path.join(fpath, 'sentiments.txt')
    data = pd.read_csv(lblpth, sep=" ", header=None)
    data.columns = ["label"]
    data[:10]

    labels_list = data['label'].tolist()

    labels = data['label'].tolist()

    labels_indices = [emo_rnk(label) for label in labels_list]

    with open(sentimentpth, 'r') as f:
        sentiments = f.read()
    # print("sentiments read read from: ", sentimentpth)

    # get rid of punctuation
    sentiments = sentiments.lower()  # lowercase, standardize
    all_text = ''.join([c for c in sentiments if c not in punctuation])

    # split by new lines and spaces
    sentiments_split = all_text.split('\n')
    all_text = ' '.join(sentiments_split)

    # create a list of words
    words = all_text.split()

    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    # vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    for ii, word in enumerate(vocab, 1):
        vocab_to_int[word] = ii

    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
    # output_size = 7
    output_size = 7
    embedding_dim = 400
    hidden_dim = 512
    n_layers = 2

    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    if checkpt:
        net.load_state_dict(torch.load(checkpt, map_location='cpu'), strict=False)
        if execute_on_gpu:
            device = torch.device("cuda")
            net.to(device)

    print("initialization done, checkpoint loaded")
    return net, execute_on_gpu


def pad_features(sentiments_ints, seq_length):
    """Return features of sentiment_ints, where each sentiment is padded with 0's
        or truncated to the input seq_length.
    """

    # getting the correct rows x cols shape
    features = np.zeros((len(sentiments_ints), seq_length), dtype=int)

    # for each review, I grab that sentiment and
    for i, row in enumerate(sentiments_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words if word in vocab_to_int.keys()])

    return test_ints


def predict(net, test_review, execute_on_gpu, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)
    
    #if test_ints has 0 word found in the training vocab, skip prediction
    if len(test_ints[0]) == 0:
        return 'Unpreditable sentence entered, please try again'

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features).long()

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size, execute_on_gpu)

    if execute_on_gpu:
        feature_tensor = feature_tensor.cuda()
    else:
        feature_tensor = feature_tensor.cpu()

    # get the output from the model
    output, h = net.forward(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    prediction = torch.round(output.squeeze())
    prediction = np.argmax(prediction.cpu().detach())
    # printing output value, before rounding

    return get_key_from_value(prediction.item())


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # lstm_out = lstm_out[-1,:,:]

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        # sig_out = self.sig(out)
        sig_out = out

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size, execute_on_gpu):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if execute_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cpu(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cpu())

        return hidden


def get_key_from_value(value):
    if len(kfv) == 0:
        for key in rnk.keys():
            v = rnk[key]
            kfv[v] = key

    result = kfv.get(value, 'unknown')

    return result


def emo_rnk(sentiment):
    return rnk.get(sentiment, 8)

def get_input_args():
    """
        gets to text to review from the command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('review', type=str,
                        help='the string to be classified in quotation marks')

    return parser


if __name__ == '__main__':
    main()
