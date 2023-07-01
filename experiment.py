import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.cuda import device
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torch.utils.data import DataLoader
# import wandb
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode, transforms
from tqdm import trange
from tqdm.contrib import tenumerate
from dataclasses import dataclass
from dataset import NpzVisionDataset
from typing import Optional


@dataclass()
class ExperimentConfig:
    epochs: int
    device: Optional[device]
    model: Optional[nn.Module]

    def __init__(self, epochs):
        self.epochs = epochs
@dataclass
class TrainingConfig(ExperimentConfig):
    optimizer: Optimizer
    lr_scheduler: LRScheduler


@dataclass
class EvaluationConfig(ExperimentConfig):
    foo: int


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    experiment_cfg = ExperimentConfig(cfg.epochs)
    experiment_cfg.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    npz_file_path = f"~/.medmnist/{cfg.dataset}mnist.npz"

    train_dataset = NpzVisionDataset(npz_file_path, to_rgb=True,
                                     transform=resnet_transform)

    experiment_cfg.dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    if cfg.model.startswith("resnet"):
        if cfg.model == "resnet18" or cfg.model == "resnet":
            model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif cfg.model == "resnet50":
            model_ft = resnet50(weights=ResNet50_Weights.DEFAULT)
        n_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(n_features, len(train_dataset.classes))

    experiment_cfg.model = model_ft.to(experiment_cfg.device)

    experiment_cfg.optimizer = Adam(experiment_cfg.model.parameters(), lr=cfg.learning_rate)
    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    experiment_cfg.lr_scheduler = MultiStepLR(experiment_cfg.optimizer, milestones=milestones, gamma=cfg.gamma)

    experiment_cfg.criterion = nn.BCEWithLogitsLoss()


    # wandb.init(
    #     project="bachelor-thesis-experiment-dev",
    #     config=OmegaConf.to_object(cfg)
    # )

    train(experiment_cfg)

    del train_dataset

    test_dataset = NpzVisionDataset(npz_file_path, split="test", to_rgb=True,
                                    transform=resnet_transform)

    experiment_cfg.dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    y_pred, y_true = evaluate(experiment_cfg)

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

    # wandb.finish()


def evaluate(cfg: ExperimentConfig):
    # testing
    cfg.model.eval()
    dataset = cfg.dataloader.dataset
    predictions_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    targets_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    with torch.no_grad():
        for i, (images, targets) in tenumerate(cfg.dataloader, desc="Evaluating", total=len(cfg.dataloader)):
            images = images.to(cfg.device)
            predictions = cfg.model(images)

            targets = targets.to(torch.float32).to(cfg.device)

            for j in range(min(cfg.dataloader.batch_size, len(predictions))):
                tensor_index = i * cfg.dataloader.batch_size + j
                predictions_t[tensor_index] = predictions[j]
                targets_t[tensor_index] = targets[j]

    return predictions_t, targets_t


def train(cfg: TrainingConfig):
    # training
    cfg.model.train()
    for _ in trange(cfg.epochs, desc="Training"):
        # TODO: https://discuss.pytorch.org/t/recording-loss-history-without-i-o/22611
        for _, (images, targets) in tenumerate(cfg.dataloader, desc="Learning", total=len(cfg.dataloader),
                                               leave=False):
            cfg.optimizer.zero_grad()

            images = images.to(cfg.device)
            predictions = cfg.model(images)

            targets = targets.to(torch.float32).to(cfg.device)
            loss = cfg.criterion(predictions, targets)

            loss.backward()
            cfg.optimizer.step()

        cfg.lr_scheduler.step()


if __name__ == '__main__':
    main()
