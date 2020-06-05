import os 
import argparse
import torch
import torchvision
from torch.autograd import Variable
from utils import set_same_seed, Generator

SEED = 0x06902059

if __name__ == '__main__':
    # Fix random seeds
    set_same_seed(SEED)

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', help = 'Path of generator checkpoint')
    parser.add_argument('--output', '-o', help = 'Path of output image')
    parser.add_argument('--z_samples', help = 'Path of z_samples')
    args = parser.parse_args()

    # Load model
    z_dim = 100
    generator = Generator(z_dim).cuda()
    generator.load_state_dict(torch.load(args.checkpoint))
    generator.eval()

    # Generate images and save the results
    num_outputs = 20

    z_sample_tensor = torch.load(args.z_samples)
    z_sample = Variable(z_sample_tensor).cuda()
    #z_sample = Variable(torch.randn(num_outputs, z_dim)).cuda()
    imgs_sample = (generator(z_sample).data + 1) / 2.0
    torchvision.utils.save_image(imgs_sample, args.output, nrow = 10)

