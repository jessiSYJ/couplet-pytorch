
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import config
from data_loader import Language, PairsLoader, get_couplets
from model import AttnDecoderRNN, EncoderRNN
from utils import as_minutes, time_since, train
from criterion import LanguageModelCriterion

SOS_token = 0
EOS_token = 1

if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)

chinese = Language(vocab_file="./couplet/vocabs")

train_pairs = get_couplets(data_dir="./couplet/train")
val_pairs = get_couplets(data_dir="./couplet/test")

train_dataloader = PairsLoader(
    chinese, train_pairs, batch_size=config.BATCH_SIZE, max_length=17)
val_dataloader = PairsLoader(chinese, val_pairs, batch_size=config.BATCH_SIZE, max_length=17)

# Initialize models
encoder = EncoderRNN(chinese.n_words, config.HIDDEN_SIZE,
                     config.NUM_LAYER, max_length=config.MAX_LENGTH+1)
decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
                         chinese.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)
if config.RESTORE:
    encoder_path = os.path.join(config.MODEL_DIR, "encoder.pth")
    decoder_path = os.path.join(config.MODEL_DIR, "decoder.pth")

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

# Move models to GPU
if config.USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.LR)
criterion = LanguageModelCriterion()#nn.NLLLoss(ignore_index=0)

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0
plot_loss_total = 0


for epoch in range(1, config.NUM_ITER + 1):

    # Get training data for this cycle
    input_index, output_index, mask_batch = next(train_dataloader.load())
    input_variable = Variable(torch.LongTensor(input_index))
    output_variable = Variable(torch.LongTensor(output_index))
    mask_variable = Variable(torch.FloatTensor(mask_batch))
    
    if config.USE_CUDA:
        input_variable = input_variable.cuda()
        output_variable = output_variable.cuda()
        mask_variable = mask_variable.cuda()
    # Run the train function
    loss = train(input_variable, output_variable, mask_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion,
                 batch_size=config.BATCH_SIZE)
    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch % config.PRINT_STEP == 0:
        print_loss_avg = print_loss_total / config.PRINT_STEP
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(
            start, epoch / config.NUM_ITER), epoch, epoch / config.NUM_ITER * 100, print_loss_avg)
        print(print_summary)

    if epoch % config.CHECKPOINT_STEP == 0:
        encoder_path = os.path.join(config.MODEL_DIR, "encoder.pth")
        decoder_path = os.path.join(config.MODEL_DIR, "decoder.pth")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)

"""
def evaluate(sentence, max_length=MAX_LENGTH):
    input_index, output_index = val_dataloader.indexes_from_sentence(sentence)
    input_variable = Variable(torch.LongTensor(input_index))
    output_variable = Variable(torch.LongTensor(output_index))
    input_variable = variable_from_sentence(chinese, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(
            2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('</s>')
            break
        else:

            decoded_words.append(chinese.index2word[ni.item()])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if config.USE_CUDA:
            decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def evaluate_randomly():
    pair = random.choice(pairs)

    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)

    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')


# evaluate_randomly()
"""
