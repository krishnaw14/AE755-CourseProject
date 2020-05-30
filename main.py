import os
import numpy as np 
import argparse

from train import train

np.random.seed(0)

parser = argparse.ArgumentParser(description='Neural Network Training')

parser.add_argument('--data', type=str, default='mnist', required=True)
parser.add_argument('--BATCH_SIZE', type=int, default=32, required=False)
parser.add_argument('--NUM_EPOCHS', type=int, default=40, required=False)
parser.add_argument('--LR', type=float, default=0.003, required=False)

parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--param_path', type=str, default=False, help='Network Parameter Path')

args = parser.parse_args()

train(args)
