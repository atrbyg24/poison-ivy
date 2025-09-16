# src/train.py
import os, argparse, yaml
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]
    model_name = params["train"]["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder("data/processed/train", transform=transform)
    val_ds = datasets.ImageFolder("data/processed/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError("Unknown model")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri("file:/app/mlruns")
    mlflow.set_experiment("toxic-plant-classification")

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "model": model_name})

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            all_labels, all_preds = [], []
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(outputs.detach().cpu().numpy())
            train_loss /= len(train_loader.dataset)
            train_acc = (np.concatenate(all_preds).argmax(axis=1) == np.concatenate(all_labels)).mean()

            model.eval()
            val_loss, val_preds, val_labels = 0.0, [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_preds.append(outputs.cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_acc = (np.concatenate(val_preds).argmax(axis=1) == np.concatenate(val_labels)).mean()

            print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pt")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
