# ## A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
# source: https://arxiv.org/pdf/1704.02971.pdf
# pytorch example: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import logging
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

logger = logging.getLogger(__file__)


class encoder(nn.Module):
    """Encoder for Dual-Stage Attention-Based RNN for timeserie prediction.

    Args:
        input_size (int): number of underlying factors.
        hidden_size (int): dimension of the hidden states.
        n_timestep (int): number of time steps.
    """

    def __init__(self, input_size, hidden_size, n_timestep):
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_timestep = n_timestep

        self.lstm_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1)

        self.attn_linear = nn.Linear(
            in_features=2 * hidden_size + n_timestep - 1, out_features=1)

    def forward(self, input_data):
        """Encoder Forward.
        Args:
            input_data (torch.tensor):
        """

        input_weighted = Variable(input_data.data.new(
            input_data.size(0), self.n_timestep - 1, self.input_size).zero_())

        input_encoded = Variable(input_data.data.new(
            input_data.size(0), self.n_timestep - 1, self.hidden_size).zero_())

        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data)  # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)

        # hidden.requires_grad = False
        # cell.requires_grad = False

        for t in range(self.n_timestep - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + n_timestep - 1)

            # Eqn. 9: Get attention weights
            # (batch_size * input_size) * 1
            x = self.attn_linear(
                x.view(-1, self.hidden_size * 2 + self.n_timestep - 1))

            # batch_size * input_size, attn weights with values sum up to 1.
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=1)

            # Eqn. 10: LSTM
            # batch_size * input_size
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(
                weighted_input.unsqueeze(0), (hidden, cell))

            hidden = lstm_states[0]
            cell = lstm_states[1]

            # Add to output tensors
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        # dimension 0 is the batch dimension
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())


class decoder(nn.Module):
    """Decoder for Dual-Stage Attention-Based RNN for timeserie prediction.

    Args:
        encoder_hidden_size (int): dimension of the encoder hidden states.
        decoder_hidden_size (int): dimension of the decoder hidden states.
        n_timestep (int): number of time steps.
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size, n_timestep):
        super(decoder, self).__init__()

        self.n_timestep = n_timestep
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(), nn.Linear(encoder_hidden_size, 1))

        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)

        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        """Decoder Forward.
        Args:
            input_encoded (torch.tensor):
            y_history (): of size (batch_size, n_timestep-1)
        """

        # input_encoded: batch_size * n_timestep - 1 * encoder_hidden_size
        # y_history: batch_size * (n_timestep-1)

        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)

        # hidden.requires_grad = False
        # cell.requires_grad = False

        for t in range(self.n_timestep - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * n_timestep * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat(
                (hidden.repeat(self.n_timestep - 1, 1, 1).permute(1, 0, 2),
                 cell.repeat(self.n_timestep - 1, 1, 1).permute(1, 0, 2),
                 input_encoded),
                dim=2)

            x = F.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.n_timestep - 1),
            dim=1)  # batch_size * n_timestep - 1, row sum up to 1

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size
            if t < self.n_timestep - 1:

                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))

                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(
                    y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))
        # logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())
