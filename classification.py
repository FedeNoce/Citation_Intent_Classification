
import time
import pandas as pd
import torch

from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')

from torch.utils.data import Dataset, DataLoader

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import clean_text, clean_numbers, _get_contractions, replace_contractions, load_glove
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from allennlp.modules import FeedForward
from allennlp.modules.elmo import Elmo, batch_to_ids
from torchtext.data.utils import get_tokenizer
import re

class MyModel(nn.Module):

    def __init__(self, elmo):
        super(MyModel, self).__init__()
        self.hidden_size = 50
        self.elmo = elmo
        drp = 0.2
        n_classes = 3
        n_section = 5
        self.lambda1 = 0.1
        self.lambda2 = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        if self.elmo:
            self.lstm = nn.LSTM(556, self.hidden_size, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(300, self.hidden_size, bidirectional=True, batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.attn = Attention(self.hidden_size*2)
        self.mlp_label = FeedForward(100, 3, [100, 20, n_classes], [nn.ReLU(), nn.ReLU(), nn.Softmax(dim=-1)], drp)
        self.mlp_section = FeedForward(100, 3, [100, 20, n_section], [nn.ReLU(), nn.ReLU(), nn.Softmax(dim=-1)], drp)
        self.mlp_worthiness = FeedForward(100, 3, [100, 20, 2], [nn.ReLU(), nn.ReLU(), nn.Softmax(dim=-1)], drp)

    def forward(self, x, x_strings, label=None, is_citation=None, section_title=None):
        embeddings_ = self.embedding(x).cuda()
        if self.elmo:
            sentences = []
            for j in range(len(x_strings)):
                tokens = tokenizer_(x_strings[j])
                if len(tokens) >= maxlen:
                    tokens = tokens[:maxlen]
                for i in range(len(tokens), maxlen):
                    tokens.append('')
                sentences.append(tokens)
            character_ids = batch_to_ids(sentences).cuda()
            elmo_embeddings = elmo(character_ids)
            elmo_embeddings_ = elmo_embeddings['elmo_representations'][0]
            embeddings_ = torch.cat((embeddings_.cuda(), elmo_embeddings_.cuda()), dim=2)
        h_lstm, _ = self.lstm(embeddings_)
        out = self.attn(h_lstm)
        if label is not None:
            out = self.mlp_label(out)
            loss = self.loss(out, label)
        if is_citation is not None:
            out = self.mlp_worthiness(out)
            loss = self.loss(out, is_citation)
            loss = loss * self.lambda1
        if section_title is not None:
            out = self.mlp_section(out)
            loss = self.loss(out, section_title)
            loss = loss * self.lambda2
        return out, loss

    def get_embeddings(self, x):
        return self.embedding(x)


def new_parameter(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in, reduction_dim=-2):
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        return weighted_sum

#Choose hyperparameters
embed_size = 300 # how big is each word vector
#max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use
batch_size = 40 # how many samples to process at once

#Set the path to csv Scicite files
data1 = pd.read_csv('/dev.csv')
data2 = pd.read_csv('/test.csv')
data3 = pd.read_csv('/train.csv')
data_section = pd.read_csv('/sections-scaffold-train.csv')
data_worthiness = pd.read_csv('/cite-worthiness-scaffold-train.csv')

data_section = pd.concat([data_section])[['text', 'section_title']]
data_worthiness = pd.concat([data_worthiness])[['text', 'is_citation']]
datas = pd.concat([data1, data2, data3])


# lower the text
datas["text"] = datas["text"].apply(lambda x: x.lower())
data_section["text"] = data_section["text"].apply(lambda x: x.lower())
data_worthiness["text"] = data_worthiness["text"].apply(lambda x: x.lower())

# Clean the text
datas["text"] = datas["text"].apply(lambda x: clean_text(x))
data_section["text"] = data_section["text"].apply(lambda x: clean_text(x))
data_worthiness["text"] = data_worthiness["text"].apply(lambda x: clean_text(x))

# Clean numbers
datas["text"] = datas["text"].apply(lambda x: clean_numbers(x))
data_section["text"] = data_section["text"].apply(lambda x: clean_numbers(x))
data_worthiness["text"] = data_worthiness["text"].apply(lambda x: clean_numbers(x))

# Clean Contractions
datas["text"] = datas["text"].apply(lambda x: replace_contractions(x))
data_section["text"] = data_section["text"].apply(lambda x: replace_contractions(x))
data_worthiness["text"] = data_worthiness["text"].apply(lambda x: replace_contractions(x))
datas = datas.astype(object).replace(np.nan, 'None')

train_X, val_X, train_y, val_y = train_test_split(datas['text'], datas['label'], test_size=0.10)

train_X_section, val_X_section, train_y_section, val_y_section = train_test_split(data_section['text'], data_section['section_title'], test_size=2)
train_X_worthiness, val_X_worthiness, train_y_worthiness, val_y_worthiness = train_test_split(data_worthiness['text'], data_worthiness['is_citation'], test_size=2)
print("Train shape : ", train_X.shape)
print("Test shape : ", val_X.shape)
print("Train shape Section: ", train_X_section.shape)
print("Train shape Worthiness: ", train_X_worthiness.shape)


tokenizer_ = get_tokenizer("basic_english")
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_X)


train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)


train_X_section = tokenizer.texts_to_sequences(train_X_section)
val_X_section = tokenizer.texts_to_sequences(val_X_section)

train_X_worthiness = tokenizer.texts_to_sequences(train_X_worthiness)
val_X_worthiness = tokenizer.texts_to_sequences(val_X_worthiness)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen, padding='post')
val_X = pad_sequences(val_X, maxlen=maxlen, padding='post')
x_train_string = tokenizer.sequences_to_texts(train_X)
x_val_string = tokenizer.sequences_to_texts(val_X)

train_X_section = pad_sequences(train_X_section, maxlen=maxlen, padding='post')
val_X_section = pad_sequences(val_X_section, maxlen=maxlen, padding='post')
x_train_string_section = tokenizer.sequences_to_texts(train_X_section)


train_X_worthiness = pad_sequences(train_X_worthiness, maxlen=maxlen, padding='post')
val_X_worthiness = pad_sequences(val_X_worthiness, maxlen=maxlen, padding='post')
x_train_string_worthiness = tokenizer.sequences_to_texts(train_X_worthiness)


## Encode the labels

le = LabelEncoder()
train_y = le.fit_transform(train_y.values)
val_y = le.transform(val_y.values)

le = LabelEncoder()
train_y_section = le.fit_transform(train_y_section.values)
val_y_section = le.transform(val_y_section.values)

le = LabelEncoder()
train_y_worthiness = le.fit_transform(train_y_worthiness.values)
val_y_worthiness = le.transform(val_y_worthiness.values)

# Load train and test in CUDA Memory
x_train = torch.tensor(train_X, dtype=torch.long).cuda()
y_train = torch.tensor(train_y, dtype=torch.long).cuda()
x_val = torch.tensor(val_X, dtype=torch.long).cuda()
y_val = torch.tensor(val_y, dtype=torch.long).cuda()

train_X_section = torch.tensor(train_X_section, dtype=torch.long).cuda()
train_y_section = torch.tensor(train_y_section, dtype=torch.long).cuda()

#
train_X_worthiness = torch.tensor(train_X_worthiness, dtype=torch.long).cuda()
train_y_worthiness = torch.tensor(train_y_worthiness, dtype=torch.long).cuda()



# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_val, y_val)

train_section = torch.utils.data.TensorDataset(train_X_section, train_y_section)

train_worthiness = torch.utils.data.TensorDataset(train_X_worthiness, train_y_worthiness)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

train_loader_section = torch.utils.data.DataLoader(train_section, batch_size=batch_size, shuffle=False)


train_loader_worthiness = torch.utils.data.DataLoader(train_worthiness, batch_size=batch_size, shuffle=False)


elmo = Elmo(options_file='/options.json', weight_file='/elmo.hdf5', num_output_representations=1).cuda()#set path to Elmo weights
train_loss = []
valid_loss = []
embedding_matrix = load_glove(tokenizer.word_index)
n_epochs = 70 #Choose epochs

#Elmo Embeddings & No scaffold
model = MyModel(elmo=True) #Choose if you want Elmo embeddings, if False only GloVe embeddings
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
model.cuda()
section = False #Choose True if you want section Scaffold
worthiness = False #Choose True if you want worthiness Scaffold
print('Start training with Elmo & No Scaffolds')
for epoch in range(n_epochs):
    start_time = time.time()
    # Set model to train configuration
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_train_string_ = x_train_string[i * batch_size:(i + 1) * batch_size]
        # Predict/Forward Pass
        y_pred, loss = model(x_batch, x_train_string_, label=y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    #First Scaffold
    if section:
        for i, (x_batch, y_batch) in enumerate(train_loader_section):
            x_train_string_section_ = x_train_string_section[i * batch_size:(i + 1) * batch_size]
        # # Predict/Forward Pass
            y_pred_2, loss = model(x_batch, x_train_string_section_, section_title=y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #Second Scaffold
    if worthiness:
        for i, (x_batch, y_batch) in enumerate(train_loader_worthiness):
            x_train_string_worthiness_ = x_train_string_worthiness[i * batch_size:(i + 1) * batch_size]
        # # Predict/Forward Pass
            y_pred_3, loss = model(x_batch, x_train_string_worthiness_, is_citation=y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Set model to validation configuration -Doesn't get trained here
    model.eval()
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_val), 3))

    for i, (x_batch, y_batch) in enumerate(valid_loader):
        x_val_string_ = x_val_string[i * batch_size:(i + 1) * batch_size]
        y_pred, _ = model(x_batch, x_val_string_, label=y_batch)
        y_pred = y_pred.detach()
        avg_val_loss += model.loss(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()
    # Check Accuracy
    #val_accuracy = sum(val_preds.argmax(axis=1) == val_y) / len(val_y)
    f1 = f1_score(val_preds.argmax(axis=1), val_y, average='macro')
    train_loss.append(avg_loss)
    valid_loss.append(avg_val_loss)
    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t f1={:.4f}  \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, f1, elapsed_time))
