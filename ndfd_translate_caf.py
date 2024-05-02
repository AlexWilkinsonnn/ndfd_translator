import argparse, os

import root
import numpy as np

import torch

def main(args):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--infile", type=str)
    parser.add_argument("--model_weights", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_arguments())
