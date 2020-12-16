import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from os.path import join, basename, splitext

from dataset import FaceDataset, load_raw_data, load_test_data
from utils import get_conf
from tqdm import tqdm

from pathlib import Path

from dataset.transforms import Normalize
from eval import evaluate
from config import Config

parser = argparse.ArgumentParser(description="Make inferences")
parser.add_argument("model", type=str, help="Path to model weights")
args = parser.parse_args()

# Load global config file
conf = get_conf("./config.yaml") 
images_train, images_val, labels_train, labels_val = load_raw_data(
    conf.train_data_path, 
    conf.train_labels_path, 
    conf.train_indices_path, 
    conf.validation_indices_path
)
images_test = load_test_data(conf.test_data_path)

# Create train data loader
transforms = [Normalize()]
train_set = FaceDataset(images_train, labels_train, transforms=transforms)
train_loader = DataLoader(train_set, batch_size=400, pin_memory=True, num_workers=12)

# Create validation data loader
validation_set = FaceDataset(images_val, labels_val, transforms=transforms)
validation_loader = DataLoader(validation_set, batch_size=400, num_workers=12)

# Create test data loader
test_set = FaceDataset(images_test, transforms=transforms)
test_loader = DataLoader(test_set, batch_size=400, num_workers=12)

from eval import mean_acc_with_thresh
model = torch.load(args.model)

pre_labels = []
for iteration, (images, _) in enumerate(validation_loader):
    out = model(images.cuda())
    out = out.cpu().detach().numpy()
    pre_labels += [out]
pre_labels = np.concatenate(pre_labels, axis=0).T[0]

# Determine the right threshold
mean_acc, th = mean_acc_with_thresh(pre_labels, labels_val, 100)
print(f"Mean accuracy : {mean_acc}, th : {th}")

# Make inferences
pre_labels = []
for iteration, images in enumerate(test_loader):
    out = model(images.cuda())
    out = out.cpu().detach().numpy()
    pre_labels += [out]
pre_labels = np.concatenate(pre_labels, axis=0).T[0]
pre_labels = pre_labels > th

with open(join("inferences", splitext(basename(args.model))[0]+"_inf.txt"), "w") as f:
    for l in pre_labels: 
        f.write(str(int(l)) + "\n")