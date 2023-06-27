import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import InterpolationMode, transforms
from tqdm.auto import tqdm, trange

from chest_dataset import ChestVisionDataset


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = ChestVisionDataset("/mnt/jbrockma/bachelor-thesis-npz/chest.npz", as_rgb=True, transform=resnet_transform)

    batch_size = 128

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)

    n_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(n_features, len(train_dataset.classes))

    model = model_ft.to(device)

    optim = Adam(model.parameters(), lr=.001)
    n_epochs = 100
    n_epochs = 2
    milestones = map(int, [.5 * n_epochs, .75 * n_epochs])
    lr_scheduler = MultiStepLR(optim, milestones=milestones, gamma=.1)

    criterion = nn.BCEWithLogitsLoss()

    # training
    model.train()
    for epoch in trange(n_epochs, desc="Training"):
        for data, targets in tqdm(train_dataloader, desc="Learning", total=len(train_dataloader), leave=False):
            optim.zero_grad()
            outputs = model(data.to(device))

            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)

            loss.backward()
            optim.step()

        lr_scheduler.step()

    test_dataset = ChestVisionDataset("/mnt/jbrockma/bachelor-thesis-npz/chest.npz", split="test", as_rgb=True, transform=resnet_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # testing
    model.eval()
    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)

            targets = targets.to(torch.float32).to(device)
            y_true = torch.cat((y_true, targets), 0)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            y_pred = torch.cat((y_pred, outputs), 0)

    y_pred = y_pred.detach().cpu().numpy().squeeze()
    y_true = y_true.detach().cpu().numpy().squeeze()

    auc = 0
    for i in range(y_true.shape[1]):
        label_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc += label_auc
    auc /= y_pred.shape[1]

    y_thresh = y_pred > .5
    acc = 0
    for i in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, i], y_thresh[:, i])
        acc += label_acc
    acc /= y_true.shape[1]

    print(f"AUC: {auc}", f"Accuracy: {acc}")


if __name__ == '__main__':
    main()

