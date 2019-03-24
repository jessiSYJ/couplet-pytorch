import os

import torch
from torch.autograd import Variable

import config
from data_loader import Language, PairsLoader
from model import AttnDecoderRNN, EncoderRNN


def indexes_from_sentence(language, sentence):
    indexes = [language.word2index[word]
               for word in sentence if word != ""]
    if len(indexes) > config.MAX_LENGTH:
        indexes = indexes[:config.MAX_LENGTH]
    # indexes.append(EOS_token)
    return indexes


def pad_sentence(sentence):
    # pad on the right
    results = [0 for i in range(config.MAX_LENGTH+1)]
    for i in range(len(sentence)):
        results[i] = sentence[i]
    return results


def load_model_param(language, model_dir):
    encoder = EncoderRNN(language.n_words, config.HIDDEN_SIZE,
                         config.NUM_LAYER, max_length=17+1)
    decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
                             language.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)

    encoder_path = os.path.join(config.MODEL_DIR, "encoder.pth")
    decoder_path = os.path.join(config.MODEL_DIR, "decoder.pth")

    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def inference(encoder, decoder, sentence, model_dir, language):
    # encoder = EncoderRNN(language.n_words, config.HIDDEN_SIZE,
    #                      config.NUM_LAYER, max_length=17+1)
    # decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
    #                          language.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)

    # encoder_path = os.path.join(config.MODEL_DIR, "encoder.pth")
    # decoder_path = os.path.join(config.MODEL_DIR, "decoder.pth")

    # encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    # decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    # encoder.eval()
    # decoder.eval()

    batch_size = 1

    input_index = indexes_from_sentence(language, sentence)
    input_index = pad_sentence(input_index)
    input_variable = torch.LongTensor([input_index])
    # Run words through encoder
    encoder_hidden, encoder_cell = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden, encoder_cell = encoder(
        input_variable, encoder_hidden, encoder_cell)

    # Prepare input and output variables
    # decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = torch.zeros(batch_size, 1).long()
    decoder_context = torch.zeros(batch_size, decoder.hidden_size)
    # Use last hidden state from encoder to start decoder
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoded_words = []
    # decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(18):
        decoder_output, decoder_context, decoder_hidden, decoder_cell, _ = decoder(
            decoder_input, decoder_context, decoder_hidden, decoder_cell, encoder_outputs)
        # decoder_attentions[di, :decoder_attention.size(
        #     2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        # decoded_words.append(chinese.index2word[ni.item()])
        if ni == 0:
            # decoded_words.append('</s>')
            break
        else:
            decoded_words.append(language.index2word[ni.item()])

        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if config.USE_CUDA:
            decoder_input = decoder_input.cuda()

    return "".join(decoded_words)


# if __name__ == "__main__":
#     chinese = Language(vocab_file="./couplet/vocabs")
#     sentence = "爱的魔力转圈圈"
#     words = inference(sentence, model_dir=config.MODEL_DIR, language=chinese)
#     print(words)
