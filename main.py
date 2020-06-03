import os
import numpy as np 
import argparse
import torch

from train import train

np.random.seed(0)
# torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Neural Network Training')

parser.add_argument('--data', type=str, default='mnist',choices=['mnist', 'taxi_time', 'devanagri'], required=True)
parser.add_argument('--optim', type=str, default='sgd', choices=['vanilla_gd', 'sgd', 'langevin_dynamics'], required=True)

parser.add_argument('--batch_size', type=int, default=32, required=False)
parser.add_argument('--num_epochs', type=int, default=40, required=False)
parser.add_argument('--lr', type=float, default=0.005, required=False)

parser.add_argument('--save_param_dir', type=str, default='saved_parameters', help='path to save learnt parameters')
parser.add_argument('--save_plots_dir', type=str, default='saved_plots', help='path to save training loss and test set evaluation plots')

parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--eval_param_dir', type=str, default=False, help='Network Parameter Path')

args = parser.parse_args()

train(args)
