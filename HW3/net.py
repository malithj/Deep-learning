import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_LENGTH = 50


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers)

    def forward(self, input, hidden):
        hidden = self.initHidden().to(self.device)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        if self.n_layers > 1:
            hidden = hidden[-1]
        return output, torch.cat([h for h in hidden], 0).unsqueeze(1)

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform_(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform_(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        self.out.weight = nn.init.xavier_uniform_(self.out.weight)
        self.attn.weight = nn.init.xavier_uniform_(self.attn.weight)
