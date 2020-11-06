import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from model import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log                  # Logging

from pathlib import Path
import copy

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--dataset', metavar='DATASET_NAME',
                    choices=DataFactory.dataset_names,
                    help='dataset to train on: ' + ' | '.join(DataFactory.dataset_names),
                    default='lh7')
parser.add_argument('--seed', type=int, metavar='N',
                    help='random number generator seed, use "-1" for random seed',
                    default=1570254494)
parser.add_argument('--num-synth', type=int, metavar='N',
                    help='number of synthetic samples to generate',
                    default=1)
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)

# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
seed = int(time.time()) if args.seed == -1 else args.seed
use_cuda = torch.cuda.is_available() and args.use_cuda

log.set_dataset_name(args.dataset)
dataset = DataFactory.instantiate(args.dataset, args.num_synth)
log.log_dataset(dataset)
log("Random seed: " + str(seed))
torch.manual_seed(seed)

hyperparameters = dataset.get_hyperparameter_set()

model_path="save/model.pt"
model = DeepGRU(dataset.num_features, dataset.num_classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(copy.deepcopy(torch.load(model_path,device)))
model.eval()


def predict_single(batch, model):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch

    if use_cuda:
        examples = examples.cuda()

    # Forward and loss computation
    outputs = model(examples, lengths)

    # Compute the accuracy
    predicted = outputs.argmax(1)
    correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    accuracy = correct / 1 * 100.0

    out = "correct" if accuracy > 0 else "nope"
    print(f"--> Predicted {dataset.idx_to_class[predicted.item()]}, expected {dataset.idx_to_class[labels[0].item()]} -> {out}")

    return accuracy


# ----------------------------------------------------------------------------------------------------------------------
def run_inference(file, model):
    """
    Runs the model on a given sample
    """

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Create data loaders
    sample_loader = dataset.get_sample_loaders(file)

    for batch in sample_loader:
        print(file)
        predicted = predict_single(batch, model)

print(dataset.idx_to_class)
for file in Path(dataset.root).glob('**/*.txt'):
    run_inference(file, model)
