
from model import EncoderRNN, AttnDecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable

USE_CUDA = False


def example_test():
    encoder_test = EncoderRNN(10, 10, 2, max_length=3)
    decoder_test = AttnDecoderRNN('general', 10, 10, 2)
    print(encoder_test)
    print(decoder_test)

    encoder_hidden = encoder_test.init_hidden(batch_size=4)
    # word_input = Variable(torch.LongTensor([[1, 2, 3]]))
    word_input = Variable(torch.LongTensor(
        [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]))
    if USE_CUDA:
        encoder_test.cuda()
        word_input = word_input.cuda()
        encoder_hidden = encoder_hidden.cuda()
    encoder_outputs, encoder_hidden = encoder_test(
        word_input, encoder_hidden)  # S B H, L B H
    print(encoder_outputs.shape, encoder_hidden.shape)
    # word_inputs = Variable(torch.LongTensor([[1, 2, 3]]))
    word_inputs = Variable(torch.LongTensor(
        [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]))
    decoder_attns = torch.zeros(4, 3, 3)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(4, decoder_test.hidden_size))

    if USE_CUDA:
        decoder_test.cuda()
        word_inputs = word_inputs.cuda()
        decoder_context = decoder_context.cuda()

    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(
            word_inputs[:, i], decoder_context, decoder_hidden, encoder_outputs)
        print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
        decoder_attns[:, i, :] = decoder_attn.squeeze(1).cpu().data


example_test()
