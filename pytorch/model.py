import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, dense_size, numeric_feature_size, output_size, lstm_layers=1,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(lstm_size, dense_size)
        self.fc2 = nn.Linear(dense_size + numeric_feature_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                      weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

            return hidden

        def forward(self, nn_input_text, nn_input_meta, hidden_state):
            batch_size = nn_input_text.size(0)
            nn_input_text = nn_input_text.long()
            embeds = self.embedding(nn_input_text)
            lstm_out, hidden_state = self.lstm(embeds, hidden_state)
            lstm_out = lstm_out[-1, :, :]
            lstm_out = self.dropout(lstm_out)
            dense_out = self.fc1(lstm_out)
            concat_layer = torch.cat((dense_out, nn_input_meta.float()), 1)
            out = self.fc2(concat_layer)
            logps = self.softmax(out)

            return logps, hidden_state


class BertTextClassifier(nn.Module):
    def __init__(self, hidden_size, dense_size, numeric_feature_size, output_size, dropout=0.1):
        super().__init__()
        self.output_size = output_size
        self.dropout = dropout

        # Use pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size + numeric_feature_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, nn_input_meta):
        all_hidden_states, all_attentions = self.bert(input_ids)[-2:]
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        dense_out = self.fc1(self.dropout(feature))
        concat_layer = torch.cat((dense_out, nn_input_meta.float()), 1)
        out = self.fc2(concat_layer)
        logps = self.softmax(out)

        return logps
