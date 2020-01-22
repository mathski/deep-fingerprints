##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Eric Wengrowski
## NYU
## Email: ew2266@nyu.edu
## Copyright (c) 2020
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for deep-fingerprints")
        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # steg training args
        train_arg = subparsers.add_parser("train",
                                    help="parser for training arguments")
        train_arg.add_argument("--epochs", type=int, default=2,
                                help="number of training epochs, default is 2")
        train_arg.add_argument("--batch-size", type=int, default=16,
                                help="batch size for training, default is 16 (~7GB)")
        train_arg.add_argument("--image-path", type=str, default="beegfs/ew2266/data/images/",
                                help="path to a folder containing another folder with all the training images")
        train_arg.add_argument("--message-path", type=str, default="beegfs/ew2266/data/messages/",
                                help="path to a folder containing another folder with all the training messages")
        train_arg.add_argument("--save-model-dir", type=str, default="models/",
                                help="path to folder where trained model will be saved.")
        train_arg.add_argument("--image-size", type=int, default=256,
                                help="size of training images, default is 256x256")
        train_arg.add_argument("--message-size", type=int, default=256,
                                help="size of message-matrix, default is 256x256")
        train_arg.add_argument("--cuda", type=int, default=1, 
                                help="set it to 1 for running on GPU, 0 for CPU")
        train_arg.add_argument("--seed", type=int, default=69, 
                                help="random seed for training")
        train_arg.add_argument("--lr", type=float, default=1e-3,
                                help="learning rate, default is 0.001")
    
    def parse(self):
        return self.parser.parse_args()