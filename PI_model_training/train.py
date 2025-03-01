"""
This module is responsible for training a deep learning model using a given dataset.
It includes functions for loading configurations, preparing datasets, creating data loaders,
instantiating the model, training the model, evaluating the model, and plotting results.
Usage:
    python train.py hyperparameters.json
"""
import re
import os
import sys
from torchinfo import summary  # type: ignore
from src.helper_functions import get_config, create_paths, get_dset_paths, get_zip_dataset, extract_zip_dataset
from src.model import Model
from src.training_functions import train, eval_model, load_best_model, plot_loss_acc, plot_confusion_matrix, torch, datasets, DataLoader, nn, transforms, lr_scheduler, plt, ConfusionMatrix
from src.path_constants import PATHS
from codecarbon import EmissionsTracker # type: ignore

def main() -> None:
    """ main(): The main function that orchestrates the training and evaluation process. """

    # Whole train tracking
    global_tracker = EmissionsTracker(output_dir="carbon_emissions", measure_power_secs=10)
    global_tracker.start()

    print("Loading config from json file.")
    config = get_config(sys.argv[2])
    config["num_workers"] = os.cpu_count()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining device: {device}\n")

    # Creating data, archives and out paths
    create_paths(PATHS)

    # Creating dataset paths
    dataset = get_dset_paths(re.split(r'[/\\]', sys.argv[1])[-1])

    print("Creating dataset ImageFolders.")
    train_set = datasets.ImageFolder(root=dataset["train_dir"],
                                        transform=transforms.ToTensor(),
                                        target_transform=None)
    test_set = datasets.ImageFolder(root=dataset["test_dir"],
                                    transform=transforms.ToTensor(),
                                    target_transform=None)

    # Get dataset classes
    classes = train_set.classes

    print("Creating dataset DataLoaders.")
    train_loader = DataLoader(dataset=train_set,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=False)

    if config["random_seed"]:
        torch.manual_seed(config["random_seed"])
        torch.cuda.manual_seed(config["random_seed"])
    print("Instanciating model.")
    model = Model(config["model"], config["input_shape"]).to(device)

    summary(model, input_size=[1] + config["input_shape"])

    # Setup loss function, optimizer and scheduler
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print("\nTraining...\n")
    model_results = train(model=model,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epochs=config["epochs"],
                        device=device,
                        model_name=config["model_name"],
                        verbose=True)

    # Plotting training and testing metrics
    plot_loss_acc(model_results)

    # Retreive best model from training
    model_best = load_best_model(model_name=config["model_name"])

    print("\nEvaluating model...")
    test_preds, model_best_results = eval_model(model=model_best,
                                                model_name=config["model_name"],
                                                dataloader=test_loader,
                                                loss_fn=loss_fn,
                                                device=device)

    print("\nBest model metrics:")
    print(model_best_results)

    # Create confusion matrix
    confmat = ConfusionMatrix(task="binary", num_classes=len(classes))
    confmat_tensor = confmat(preds=test_preds,
                            target=torch.tensor(test_set.targets))

    # Plot confusion matrix
    _, _ = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=classes,
        figsize=(10, 7)
    )
    plt.show()

    # Whole train tracking
    total_emissions = global_tracker.stop()
    print(f"Total emissions for training: {total_emissions:.4f} kg CO2eq")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py data/<dataset_folder> config.json")
        exit()
    main()