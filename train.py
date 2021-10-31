import os
import argparse
import datetime
import math

from config import Config

import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    config = Config()
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    # Sửa đổi biến đổi hình ảnh - image augmentation
    transform = transforms.Compose([ 
        transforms.RandomCrop(config.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(config.img_path,
                            config.train_path,
                            vocab,
                            transform,
                            config.tokenizer,
                            config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers)


    # Build the models
    encoder = EncoderCNN(config.embed_size).to(device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers).to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr = config.learning_rate)
    else:
        raise NotImplementedError("Other optimizers not implemented.")

    # encoder's state_dict
    for param_tensor in encoder.state_dict():
        print(f"{param_tensor} \t {encoder.state_dict()[param_tensor].size()}")
    # decoder's state_dict
    for param_tensor in decoder.state_dict():
        print(f"{param_tensor} \t {decoder.state_dict()[param_tensor].size()}")
    # optimizer's state_dict
    for var_name in optimizer.state_dict():
        print(f"{var_name} \t {optimizer.state_dict()[var_name]}")

    # Let's train the model
    best_valid_loss = float('inf')

    # Xác định số bước huấn luyện trong một epoch
    total_step = len(data_loader)

    for epoch in range(config.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # https://stackoverflow.com/questions/61988776/how-to-calculate-perplexity-for-a-language-model-using-pytorch
            # Print log info
            if i % config.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, config.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save each model's components checkpoints
            if (i+1) % config.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    config.model_path, 'decoder-{}-{}-{}.ckpt'.format(epoch+1, i+1, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
                torch.save(encoder.state_dict(), os.path.join(
                    config.model_path, 'encoder-{}-{}-{}.ckpt'.format(epoch+1, i+1, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))

                # save model checkpoint
                torch.save({
                            'epoch': epoch+1,
                            'encoder_state_dict': encoder.state_dict(),
                            'decoder_state_dict': decoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                            }, './outputs/model.pth')


if __name__ == "__main__":
    main()