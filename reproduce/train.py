import os
import argparse
import torch
import torchvision
import torch.nn as nn
#import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import set_same_seed, Generator, Discriminator, get_dataset

torch.cuda.set_device(0)
SEED = 0x06902059

if __name__ == '__main__':
    # Fix random seed
    set_same_seed(SEED)

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type = int, help = 'GPU device to be trained on')
    parser.add_argument('--checkpoint', '-c', help = 'Path of checkpoint')
    parser.add_argument('--input_dir', '-i', help = 'Directory of input images')
    args = parser.parse_args()
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

   
    # Hyperparameters
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    num_epochs = 10

    # Build models
    generator = Generator(in_dim = z_dim).cuda()
    discriminator = Discriminator(3).cuda()
    generator.train()
    discriminator.train()

    # Criterion
    criterion = nn.BCELoss()

    # Optimizer 
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr = lr, betas = (0.5, 0.999))

    # Dataset & dataloader 
    dataset = get_dataset(args.input_dir)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    # for logging
    z_sample = Variable(torch.randn(100, z_dim)).cuda()

    for e, epoch in enumerate(range(num_epochs)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = generator(z)

            # Label
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # Discriminator
            r_logit = discriminator(r_imgs.detach())
            f_logit = discriminator(f_imgs.detach())
        
            # Compute loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # Update model
            discriminator.zero_grad()
            loss_D.backward()
            optimizer_discriminator.step()

            """ train G """
            # Leaf
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = generator(z)

            # Discriminator
            f_logit = discriminator(f_imgs)
        
            # Compute loss
            loss_G = criterion(f_logit, r_label)

            # Update model
            generator.zero_grad()
            loss_G.backward()
            optimizer_generator.step()

            # Log
            print(f'\rEpoch [{epoch + 1}/{num_epochs}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')

    torch.save(generator.state_dict(), args.checkpoint)
    
