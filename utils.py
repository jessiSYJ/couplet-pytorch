import os
import random
import time
import math
import torch
from torch.autograd import Variable
import config

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))




def train(input_variable, target_variable, mask_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, batch_size):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    target_length = target_variable.size()[1]

    # Run words through encoder
    encoder_hidden, encoder_cell = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_variable, encoder_hidden, encoder_cell)

    # Prepare input and output variables
    decoder_input = Variable(torch.zeros(batch_size, 1).long())
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    # Use last hidden state from encoder to start decoder
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    decoder_outputs = []
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, decoder_cell, encoder_outputs)
        # loss += criterion(decoder_output, target_variable[:, di])
        decoder_outputs.append(decoder_output)
        decoder_input = target_variable[:, di]  # Next target is next input
    decoder_predict = torch.cat(decoder_outputs, 1).view(batch_size, target_length, -1)
    loss = criterion(decoder_predict, target_variable, mask_variable)
    """
    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < config.TEACHER_RATIO
    if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[:, di])
            decoder_input = target_variable[:, di]  # Next target is next input
    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[:, di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[:, 0]
            decoder_input = ni
            # decoder_input = Variable(torch.LongTensor([[ni]]))
            if config.USE_CUDA:
                decoder_input = decoder_input.cuda()
    """
    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.CLIP)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.CLIP)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() # / target_length
