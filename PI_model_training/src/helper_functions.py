import matplotlib.pyplot as plt
import random
import zipfile
from PIL import Image
import json
from typing import List, Dict

from src.path_constants import *


def get_config(json_file):
    with open(json_file, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters


def get_dset_paths(dset_name):
    return {"name": dset_name,
            "dir": PATHS["data"] / dset_name,
            "train_dir": PATHS["data"] / dset_name / "train",
            "test_dir": PATHS["data"] / dset_name / "test"}


def get_zip_dataset(dset_name, dset_url):
    if (PATHS["archives"] / f"{dset_name}.zip").is_file():
        print(f"{dset_name}.zip is already dowloaded, skipping")
        return
    os.system(f"curl -L -o {PATHS['archives']}/{dset_name}.zip {dset_url}")


def extract_zip_dataset(dset_name):
    if (PATHS["data"] / dset_name).is_dir():
        print(f"{dset_name} dataset already exists, skipping")
        return
    with zipfile.ZipFile(PATHS["archives"] / f"{dset_name}.zip") as zip_ref:
        zip_ref.extractall(PATHS["data"] / dset_name)


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are ", end='')
        if len(dirnames) > 0:
            print(f"{len(dirnames)} directories ", end='')
            if len(filenames) > 0:
                print("and ", end='')
        if len(filenames) > 0:
            print(f"{len(filenames)} images ", end='')
        print(f"in {dirpath}.")


def secs_to_hmsms(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{'0' if hours < 10 else ''}{hours}:{'0' if minutes < 10 else ''}{minutes}:{'0' if seconds < 10 else ''}{seconds}.{milliseconds}"


def plot_random_images(dset_infos, dir: str):
    if dir not in ("REAL", "FAKE"):
        print("dir must be REAL or FAKE")
        return
    image_path_list = list(dset_infos["dir"].glob(f"*/{dir}/*.jpg"))
    random_image_paths = random.sample(image_path_list, 16)
    fig = plt.figure(figsize=(8, 8))
    plt.axis(False)
    rows, cols = 4, 4
    for i, image_path in enumerate(random_image_paths):
        fig.add_subplot(rows, cols, i+1)
        plt.tight_layout()
        image_class = image_path.parent.stem
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(image_class)
        plt.axis(False)
    plt.show()


def plot_loss_acc(results: Dict[str, List[float]]):
    """
    Plots training curves of a results dictionary.
    """
    # Get the loss values of the results dictionary
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary
    train_acc = results["train_acc"]
    test_acc = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(train_loss))

    # Setup a plot
    plt.figure(figsize=(12, 6))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, test_acc, label="Test acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()