import os, argparse, yaml
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
)
from torch.utils.data import DataLoader
import numpy as np

MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
    "squeezenet1_0": (squeezenet1_0, SqueezeNet1_0_Weights.DEFAULT)
}

def build_model(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")
    
    model_fn, weights = MODELS[model_name]
    model = model_fn(weights=weights)

    if model_name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "squeezenet1_0":
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=1)

    return model

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]
    model_name = params["train"]["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([T.Resize((224,224)), 
                           T.ToTensor(), 
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])
    
    train_ds = datasets.ImageFolder("data/processed/train", transform=transform)
    val_ds = datasets.ImageFolder("data/processed/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri("file:/app/mlruns")
    mlflow.set_experiment("toxic-plant-classification")

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "model": model_name})

        for epoch in range(epochs):
            model.train()
            train_loss, all_labels, all_preds = 0.0, [], []
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
