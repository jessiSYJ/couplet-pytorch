import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import config


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, \
                 n_layers=1, max_length=config.MAX_LENGTH):

        super(EncoderRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = n_layers
        self._max_length = max_length
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=False, num_layers=n_layers)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden, cell):

        word_inputs = torch.transpose(word_inputs, 1, 0) # T x N
        embedded = self.embedding(word_inputs).view(self._max_length, -1, self._hidden_size) # T x N x H
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        return output, hidden, cell

    def init_hidden(self, batch_size):

        hidden = Variable(torch.zeros(self._num_layers, batch_size, self._hidden_size)) # when bidirect=true, *2
        cell = Variable(torch.zeros(self._num_layers, batch_size, self._hidden_size)) 
        if config.USE_CUDA:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return hidden, cell


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs): # B x N*2 || S x B x N*2
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len, batch_size))  # S x B 
        if config.USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim=0).transpose(0, 1).unsqueeze(1) # B x 1 x S

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output) 
            energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2)) # batch inner product
            return energy.squeeze()

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # self.batch_size = batch_size

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size * 2, hidden_size, 
            bidirectional=False, num_layers=n_layers, dropout=dropout_p)
        # self.gru = nn.GRU(hidden_size * 2, hidden_size,
        #                   n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, last_cell, encoder_outputs):
        word_embedded = self.embedding(
            word_input).view(1, -1, self.hidden_size)  # S=1 x B x N

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2) # 1 x B x N*2
        rnn_output, (hidden, cell) = self.rnn(rnn_input, (last_hidden, last_cell)) # 1 x B x N , lay_num x B x N 
        rnn_output = rnn_output.squeeze(0)  #  B x N

        attn_weights = self.attn(rnn_output, encoder_outputs) # B x 1 x S
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, cell, attn_weights
