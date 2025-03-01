import json
from time import time
from tqdm.auto import tqdm
import copy

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchinfo
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from codecarbon import EmissionsTracker # type: ignore


from .helper_functions import *


def get_train_set(dset_infos, transform, target_transform):
    return datasets.ImageFolder(root=dset_infos["train_dir"],
                                    transform=transform,
                                    target_transform=target_transform)

def get_test_set(dset_infos, transform, target_transform):
    return datasets.ImageFolder(root=dset_infos["test_dir"],
                                    transform=transform,
                                    target_transform=target_transform)


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle)


# Create train_step()
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put the model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloder data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X, y = X.to(device), y.type(torch.float).to(device)

        # 1. Forward pass
        y_logits = model(X).squeeze()

        # 2. Calculate the loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred = y_logits.sigmoid().round()
        train_acc += (y_pred == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


# Create a test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.type(torch.float).to(device)

            # 1. Forward pass
            test_logits = model(X).squeeze()

            # 2. Calculate the loss
            test_loss += loss_fn(test_logits, y).item()

            # Calulate the accuracy
            test_pred = test_logits.sigmoid().round()
            test_acc += (test_pred == y).sum().item() / len(test_pred)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def save_metrics(model_name, metrics):
    with open(PATHS["out"] / f"{model_name}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
                

def load_metrics(model_name):
    with open(PATHS["out"] / f"{model_name}_metrics.json", 'r') as f:
        metrics = json.load(f)
    return metrics


# Create a train function that takes in various model parameters + optimizer + loss function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device,
          epochs: int,
          model_name: str,
          verbose: bool=True):

    # Create empty results dictionary
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    if verbose:
        # Start the timer
        start_time = time()

    best_test_acc = 0.

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        print(f"\nEpoch {epoch + 1}/{epochs}\n{'-' * 20}")

        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        scheduler.step()

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)


        if test_acc > best_test_acc:
            torch.save(model, PATHS["out"] / f"{model_name}.pt")
            best_test_acc = test_acc

        # Print loss and acc
        if verbose:
            tqdm.write(f"Epoch {epoch + 1}:")
            tqdm.write(f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}%")
            tqdm.write(f"Test loss:  {test_loss:.4f} | Test acc:  {test_acc*100:.2f}%\n")

        # Update results dictionary
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_acc"].append(test_acc)
        save_metrics(model_name, metrics)
       
    if verbose:
        # End the timer and print out how long it took
        end_time = time()
        print(f"Total training time: {secs_to_hmsms(end_time - start_time)}")

    # Return the results at the end of the epochs
    return metrics


def load_best_model(model_name):
    model_best = torch.load(PATHS["out"] / f"{model_name}.pt")
    return model_best


def eval_model(model: torch.nn.Module,
               model_name,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
    """
    Returns a dictionary containing the results of model predicting on data loader.
    """
    
    test_preds = []
    
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop throuogh DataLoader batches
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Send data to the target device
            X, y = X.to(device), y.type(torch.float).to(device)

            # 1. Forward pass
            test_logits = model(X).squeeze()

            # 2. Calculate the loss
            test_loss += loss_fn(test_logits, y).item()

            # Calulate the accuracy
            test_pred = test_logits.sigmoid().round()
            test_acc += (test_pred == y).sum().item() / len(test_pred)
            
            test_preds.append(test_pred.cpu())

        # Adjust metrics to get average loss and accuracy per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        test_preds_tensor = torch.cat(test_preds)

    return  test_preds_tensor, {"model_name": model_name,
                                "model_loss": test_loss,
                                "model_acc": test_acc}