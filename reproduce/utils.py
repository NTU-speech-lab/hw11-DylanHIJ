import cv2
import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)
    ]) 
    dataset = FaceDataset(fnames, transform)
    return dataset

def set_same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim = 64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias = False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding = 2, output_padding = 1),
            nn.Tanh())
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim = 64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        self.apply(weights_init)    

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

class SNGANDiscriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim = 64):
        super(SNGANDiscriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_dim, out_dim, 5, 2, 2)),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        
        self.ls = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, dim, 5, 2, 2)), 
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            #nn.Dropout(0.5),
            conv_bn_lrelu(dim * 2, dim * 4),
            #nn.Dropout(0.5),
            conv_bn_lrelu(dim * 4, dim * 8),
            #nn.Dropout(0.5),
            spectral_norm(nn.Conv2d(dim * 8, 1, 4)),
            nn.Sigmoid())
        
        self.apply(weights_init)        

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
