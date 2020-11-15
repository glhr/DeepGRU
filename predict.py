import argparse
import numpy as np
import time

import torch
import torch.nn as nn

from DeepGRU.model import DeepGRU
from DeepGRU.dataset.datafactory import DataFactory
from DeepGRU.utils.average_meter import AverageMeter  # Running average computation
from DeepGRU.utils.logger import log                  # Logging

from DeepGRU.utils.utils import get_path_from_root

from pathlib import Path
import copy

seed = 1570254494
use_cuda = torch.cuda.is_available()
dataset = 'lh7'
num_synth = 1

log.set_dataset_name(dataset)
dataset = DataFactory.instantiate(dataset, num_synth)
log.log_dataset(dataset)
log("Random seed: " + str(seed))
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameters = dataset.get_hyperparameter_set()

def get_model(path="save/model.pt"):
    model_path = get_path_from_root(path)
    print(model_path)
    model = DeepGRU(dataset.num_features, dataset.num_classes)
    state_dict = torch.load(model_path,device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    state_dict = new_state_dict
    # load params
    model.load_state_dict(state_dict)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model

model = get_model()


def predict_single(batch, model, eval=False):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch

    if use_cuda:
        examples = examples.cuda()

    # Forward and loss computation
    output = nn.functional.softmax(model(examples, lengths),dim=1)

    # Compute the accuracy
    ret, predicted = torch.max(output.data, 1)
    top_p, top_class = output.topk(1, dim = 1)
    conf = top_p.item()
    cls = dataset.idx_to_class[top_class.item()]

    if eval:
        correct = (predicted == labels).sum().item()
        curr_batch_size = labels.size(0)
        accuracy = correct / 1 * 100.0

        out = "correct" if accuracy > 0 else "nope"
        # print(f"--> Predicted {cls}, expected {dataset.idx_to_class[labels[0].item()]} -> {out}")
        return {'predicted': cls, 'expected': dataset.idx_to_class[labels[0].item()]}
    else:
        print(f"--> Predicted {cls} (conf {conf})")
        return {'label': cls, 'conf': conf}


# ----------------------------------------------------------------------------------------------------------------------
def run_inference(input):
    global model
    """
    Runs the model on a given sample
    """

    # Create data loaders
    sample_loader = dataset.get_sample_loaders(input)
    predicted = []
    for batch in sample_loader:
        if isinstance(input,str): print(file)
        predicted.append(predict_single(batch, model))

    return predicted

if __name__ == 'main':
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
    dataset = args.dataset
    num_synth = args.num_synth

    print(dataset.idx_to_class)
    for file in Path(dataset.root).glob('**/*.txt'):
        run_inference(str(file))
