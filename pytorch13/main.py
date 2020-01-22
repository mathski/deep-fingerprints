##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Eric Wengrowski
## NYU
## Email: ew2266@nyu.edu
## Copyright (c) 2020
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch

import nets
import utils
import options

from option import Options


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

def train():
    check_paths(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # construct data loader
    dataset = MyDataset(args.image_paths, args.message_paths)

    # construct neural networks
    encoder = nets.CSFE()
    decoder = nets.CSFD()
    if args.cuda:
        encoder.cuda()
        decoder.cuda()

    # construct optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # define loss
    l2 = torch.nn.MSELoss()

    # training loop
    for e in range(args.epochs):
        
        for img, code in dataset:

            # encode image and code
            enc = encoder(img, code)

            # channel

            # decode encoded image
            output = decoder(enc)

            # calculate loss
            ber_loss = l2(output, code)
            img_loss = l2(enc, img)
            total_loss = ber_loss + img_loss

            # occasionally print
            print(total_loss)

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

class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        return image, mask

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    main()