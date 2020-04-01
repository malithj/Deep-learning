import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torchtext
import argparse
import sys
import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from skimage import io, transform
from torchtext.data.metrics import bleu_score

from net import EncoderRNN, AttnDecoderRNN, DecoderRNN
from text_mapping import TextMappingDataset


teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
MAX_LENGTH= 50


def parse():
    parser = argparse.ArgumentParser(description='Simple Neural Network Model')
    subparsers = parser.add_subparsers(help='select neural network mode', dest='mode')
    parser_train = subparsers.add_parser('train', help='training mode')
    parser_test = subparsers.add_parser('test', help='testing mode')
    parser_translate = subparsers.add_parser('translate', help='translate mode')
    args = parser.parse_args()
    var = vars(args)
    if 'mode' not in var:
        mode = ''
    else:
        mode = var['mode']
    if mode == 'test':
        print("Testing mode activated")
    elif mode == 'train':
        print("Training mode activated")
    elif mode == 'translate':
        print("Translate mode activated")
    else:
        raise Exception("Please check input arguments")
    return mode


def load_train_data():
    e_url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en"
    f_url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi"
    torchtext.utils.download_from_url(e_url, root='./data', overwrite=False)
    torchtext.utils.download_from_url(f_url, root='./data', overwrite=False)
    train_data =  TextMappingDataset(english_file='./data/train.en',
                                     foriegn_file='./data/train.vi',
                                     root_dir='./data')
    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=2)
    return train_data_loader, train_data.input_lang, train_data.output_lang


def load_test_data():
    test_e_url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en"
    test_f_url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi"
    torchtext.utils.download_from_url(test_e_url, root='./data', overwrite=False)
    torchtext.utils.download_from_url(test_f_url, root='./data', overwrite=False)
    test_data =  TextMappingDataset(english_file='./data/tst2013.en',
                                     foriegn_file='./data/tst2013.vi',
                                     root_dir='./data')
    test_data_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=2)
    return test_data_loader, test_data.input_lang, test_data.output_lang


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden().to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(min(MAX_LENGTH, input_length)):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = decoder.initHidden().to(device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_itrs(trainloader, encoder, decoder, device, n_iters, print_every=1000, learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every

    ENC_PATH = './model/encoder.pth'
    DEC_PATH = './model/decoder.pth'
    
    encoder.init_weight()
    decoder.init_weight()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    min_loss = np.inf
    print("Loading Model....")
    print("{0:>4s}  {1:>15s} {2:>12s}".format("Loop", "Progress", "Train Loss"))
    for loop in range(n_iters):
        for iter, data in enumerate(trainloader, 0):
            input_tensor = data['english_txt'][0].to(device)   # adjust batch size
            target_tensor = data['foriegn_txt'][0].to(device)  # adjust batch size
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, device)
            print_loss_total += loss
        
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / (loop * len(trainloader) + print_every)
                print_loss_total = 0
                if min_loss > print_loss_avg:
                        torch.save(encoder.state_dict(), ENC_PATH)
                        torch.save(decoder.state_dict(), DEC_PATH)
                        min_loss = print_loss_avg
                print("{0:>4d} {1:>15.2f}% {2:>12f}".format(loop * len(trainloader) + iter, ((loop * len(trainloader) + iter) / (n_iters * len(trainloader))) * 100, print_loss_avg))


def evaluate(encoder, decoder, input_tensor, device, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden().to(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = decoder.initHidden().to(device)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def test(testloader, encoder, decoder, device, input_lang, output_lang):
    _bleu_score = 0
    _target_tensor_list = []
    _decoded_output_list = []
    for iter, data in enumerate(testloader, 0):
        input_tensor = data['english_txt'][0].to(device)
        target_tensor = data['foriegn_txt'][0].to(device)
        
        input_length = input_tensor.size()[0]
        if input_length > MAX_LENGTH:
            continue
        decoded_output, attns = evaluate(encoder, decoder, input_tensor, device)
        _decoded_output_list.append(output_lang.sentenceFromTensor(decoded_output))
        _intermediate_target = target_tensor.tolist()
        _intermediate_target = [x[0] for x in _intermediate_target]
        _target_tensor_list.append(output_lang.sentenceFromTensor(_intermediate_target))
    _bleu_score = bleu_score(_target_tensor_list, _decoded_output_list, max_n=1, weights=[1])
    print("Average BLEU: {0:15.6f}".format(_bleu_score))


def translate(encoder, decoder, device, input_lang, output_lang):
    print("Type exit to deactivate translate mode")
    print("Please enter text to be translated: ", end="")
    while(True):
        input_str_ = str(input())
        if input_str_.strip() == "exit":
            break
        input_tensor = input_lang.tensorFromSentence(input_str_).to(device)
        output_tensor, _ = evaluate(encoder, decoder, input_tensor, device)
        output_str_ = output_lang.sentenceFromTensor(output_tensor)
        print(' '.join(output_str_))
        print("Please enter text to be translated: ", end="")


if __name__ == '__main__':
    if not os.path.exists('model'):
        try:
            os.makedirs('model')
        except OSError as e:
            raise Exception("Cannot create directory")
    if not os.path.exists('./data'):
        try:
            os.makedirs('./data')
        except OSError as e:
            raise Exception("Cannot create directory")
    mode = parse()
    ENC_PATH = './model/encoder.pth'
    DEC_PATH = './model/decoder.pth'
    hidden_size = 256
    n_layers=3
    n_iters=50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, input_lang, output_lang = load_train_data()
    if mode == 'train':
        encoderRNN = EncoderRNN(input_lang.n_words, hidden_size, device, n_layers).to(device)
        decoderRNN = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        train_itrs(train_loader, encoderRNN, decoderRNN, device,  n_iters=n_iters, print_every=1000, learning_rate=0.001)
    elif mode == 'test':
        test_loader,_,_ = load_test_data()
        test_loader.dataset.input_lang = input_lang
        test_loader.dataset.output_lang = output_lang
        encoderRNN = EncoderRNN(input_lang.n_words, hidden_size, device, n_layers).to(device)
        decoderRNN = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        encoderRNN.load_state_dict(torch.load(ENC_PATH))
        decoderRNN.load_state_dict(torch.load(DEC_PATH))
        test(test_loader, encoderRNN, decoderRNN, device, input_lang, output_lang)
    else:
        encoderRNN = EncoderRNN(input_lang.n_words, hidden_size, device, n_layers).to(device)
        decoderRNN = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        encoderRNN.load_state_dict(torch.load(ENC_PATH))
        decoderRNN.load_state_dict(torch.load(DEC_PATH))
        translate(encoderRNN, decoderRNN, device, input_lang, output_lang)
