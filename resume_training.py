import os
import argparse
import datetime

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

    # load the model checkpoint
    checkpoint = torch.load('outputs/model.pth')
    # load model weights state_dict
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    # load trained optimizer state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Previously trained optimizer state_dict loaded...')
    old_epochs = checkpoint['epoch']
    # load the criterion
    criterion = checkpoint['loss']
    print('Trained model loss function loaded...')
    print(f"Previously trained for {old_epochs} number of epochs...")
    # train for more epochs
    epochs = config.new_epochs
    print(f"Train for {epochs} more epochs...")
    total_epochs = old_epochs + epochs

    # Let's train the model
    total_step = len(data_loader)
    for epoch in range(old_epochs, total_epochs, 1):
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

            # Print log info
            if i % config.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, total_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save each model component checkpoints
            if (i+1) % config.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    config.model_path, 'decoder-{}-{}-{}.ckpt'.format(epoch+1, i+1, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))))
                torch.save(encoder.state_dict(), os.path.join(
                    config.model_path, 'encoder-{}-{}-{}.ckpt'.format(epoch+1, i+1, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))))

                # save model checkpoint
                torch.save({
                            'epoch': epoch+1,
                            'encoder_state_dict': encoder.state_dict(),
                            'decoder_state_dict': decoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                            }, 'outputs/model.pth')



if __name__ == "__main__":
    main()