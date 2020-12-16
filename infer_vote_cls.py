import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from os.path import join, basename, splitext

from dataset import FaceDataset, load_raw_data
from utils import get_conf
from tqdm import tqdm

from pathlib import Path

from dataset.transforms import Normalize
from eval import evaluate
from config import Config

parser = argparse.ArgumentParser(description="Make inferences with several models and apply logistic regression")
parser.add_argument("models", type=str, nargs="+", help="Path to models weights")
args = parser.parse_args()

# Load global config file
conf = get_conf("./config.yaml") 
images_train, images_test, labels_train, labels_test = load_raw_data(
    conf.train_data_path, 
    conf.train_labels_path, 
    conf.train_indices_path, 
    conf.validation_indices_path
)

# Create train data loader
transforms = [Normalize()]
train_set = FaceDataset(images_train, labels_train, transforms=transforms)
train_loader = DataLoader(train_set, batch_size=400, pin_memory=True, num_workers=12)

# Create validation data loader
validation_set = FaceDataset(images_test, labels_test, transforms=transforms)
validation_loader = DataLoader(validation_set, batch_size=400, num_workers=12)

from eval import mean_accuracy
from vote_classifier import VoteClassifier

classifier = VoteClassifier([torch.load(model_path) for model_path in args.models])
classifier.fit_lr(train_loader)
output = classifier.forward(validation_loader)
mean_acc = mean_accuracy(output, labels_test)
print(f"Mean accuracy : {mean_acc}")