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

parser = argparse.ArgumentParser(description="Train a model on face images")
parser.add_argument("param_file", type=str, help="Parameters file path")
parser.add_argument("--mlflow", action="store_true", help="Enables logging to MLflow")
args = parser.parse_args()

# Load global config file
conf = get_conf("./config.yaml") 
images_train, images_test, labels_train, labels_test = load_raw_data(
    conf.train_data_path, 
    conf.train_labels_path, 
    conf.train_indices_path, 
    conf.validation_indices_path
)

# Load run config file
params = Config.from_yaml(args.param_file)

# Create train data loader
transforms = params.transforms()
if not isinstance(transforms, list):
    transforms = [transforms]
train_set = FaceDataset(images_train, labels_train, transforms=transforms)
train_loader = DataLoader(train_set, batch_size=params.training.batch_size(), pin_memory=True, num_workers=12)

# Create validation data loader
validation_set = FaceDataset(images_test, labels_test, transforms=transforms)
validation_loader = DataLoader(validation_set, batch_size=400, num_workers=12)

# Instantiate model
model = params.model().cuda()

# Loss and optimizer
lr = params.training.optimizer.params.lr()
weight = torch.tensor(params.training.weight()).float().cuda() # Get weights of each class for the loss
entropy_loss = params.training.loss(reduce=False)
optimizer = params.training.optimizer(params=model.parameters()) 
scheduler = params.training.scheduler(optimizer=optimizer) 

# Setup MLflow
if args.mlflow:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Idemia faces")
    mlflow.start_run(run_name=model.__class__.__name__)
    mlflow.log_artifact(args.param_file)
    mlflow.log_param("lr", lr)
    mlflow.log_param("batch_size", params.training.batch_size())
    mlflow.log_param("Optimizer", params.training.optimizer.class_name)

# Training loop
epochs = params.training.epochs()
losses = []
progress_bar = tqdm(total=epochs * len(train_loader), ascii=True)
for epoch in range(epochs):
    for iteration, (images, labels) in enumerate(train_loader):
        model.train() 
        progress_bar.update(1)
        # Train
        out = model(images.cuda())
        loss = entropy_loss(out, labels.cuda().unsqueeze(1))
        loss = torch.mean(loss * weight)
        losses += [loss.item()]
        progress_bar.set_description(f"Epoch : {epoch} loss : {loss.item():.4}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log loss to MLflow
        if iteration % int(len(train_loader) / 2) - 1 == 0 and iteration > 0:
            mean_loss = np.mean(losses)
            if args.mlflow: 
                mlflow.log_metric("Loss", mean_loss, step=epoch)
            losses = []

            # Validation
            model.eval()
            auc, precision = evaluate(model, validation_loader, labels_test, progress_bar=progress_bar)
            if args.mlflow:
                mlflow.log_metric("Val auc", auc)
                mlflow.log_metric("Val mean acc", precision)
    
    scheduler.step()

    # Save checkpoint
    model_dir = join("./checkpoints/", splitext(basename(args.param_file))[0])
    Path(model_dir).mkdir(exist_ok=True)
    if epoch % 2 == 0 and epoch > 0:
        torch.save(model, join(model_dir, f"e{epoch}.pth")) 

if args.mlflow:
    mlflow.end_run()