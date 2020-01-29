##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Eric Wengrowski
## NYU
## Email: ew2266@nyu.edu
## Copyright (c) 2020
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import glob
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

import nets
import utils
from options import Options

from barbar import Bar

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, target_paths, image_transform=None, message_transform=None, train=True):
        self.image_paths = glob.glob(image_paths)
        self.target_paths = glob.glob(target_paths)
        self.image_transform = image_transform
        self.message_transform = message_transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        message = Image.open(self.target_paths[index])
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.message_transform:
            message = self.message_transform(message)

        image = transforms.ToTensor()(image)
        message = transforms.ToTensor()(message)

        return image, message

    def __len__(self):
        return len(self.image_paths)

def main():
    # Print PyTorch Version
    print('\nUsing PyTorch Version:------>', torch.__version__)

    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    elif args.subcommand == 'train':
        # Train watermarking network
        train(args)

    else:
        raise ValueError('Unknown experiment type')

def train(args):
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # construct data loader
    dataset = MyDataset(args.image_path, args.message_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=False)

    # construct neural networks
    encoder = nets.CSFE()
    decoder = nets.CSFD()
    if args.cuda:
        encoder.cuda()
        decoder.cuda()

    # construct optimizer
    params = list(encoder.parameters())+list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # define loss
    l2 = torch.nn.MSELoss()

    # training loop
    for e in range(args.epochs):
        
        for img, code in Bar(loader):

            # encode image and code
            enc = encoder(img, code)

            # channel
            #nothing for now

            # decode encoded image
            output = decoder(enc)

            # calculate loss
            ber_loss = l2(output, code)
            img_loss = l2(enc, img)
            total_loss = ber_loss + img_loss

            # occasionally print
            #print(total_loss)

            #backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # save trained model

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()