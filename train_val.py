from __future__ import print_function
from ssds_train import train_model
import argparse
import sys
from utils.config import cfg_from_file

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='config_file',default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    train_model()

if __name__ == '__main__':
    train()

