import copy
import datetime
import logging
import os
import time
from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn, device
from torch.nn.modules import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, get_model_weights, \
    WeightsEnum
from torchvision.transforms import transforms
from tqdm import trange
from tqdm.contrib import tenumerate

from dataset import create_dataset_init
from tuning import get_tuned_model


@dataclass(init=False)
class ExperimentConfig:
    device: device
    model: nn.Module
    criterion: Module
    data_loader: DataLoader


@dataclass
class TrainingConfig(ExperimentConfig):
    optimizer: Optimizer
    lr_scheduler: LRScheduler

    def __init__(self, data_loader: DataLoader, experiment_cfg: ExperimentConfig):
        for attr_name, attr_value in vars(experiment_cfg).items():
            setattr(self, attr_name, attr_value)
        self.data_loader = data_loader


log = logging.getLogger(__name__)


def create_dataloader_init(batch_size):
    def init(dataset: Dataset, **kwargs):
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True, **kwargs)

    return init


def get_default_model_weights(name: str) -> WeightsEnum:
    weights = get_model_weights(name)
    return getattr(weights, "DEFAULT")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Using config: {cfg}")

    experiment_cfg = ExperimentConfig()

    experiment_cfg.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Using device: {experiment_cfg.device}")

    imagenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    npz_file_path = os.path.join(cfg.dataset.path, f"{cfg.dataset.name}.npz")

    init_dataset = create_dataset_init(cfg.dataset.name, npz_file_path, transform=imagenet_transform)

    log.info("Loading training dataset")
    train_dataset = init_dataset()
    log.info("Training dataset loaded")

    init_data_loader = create_dataloader_init(cfg.batch_size)

    train_data_loader = init_data_loader(train_dataset, shuffle=True)

    log.info("Loading validation dataset")
    val_dataset = init_dataset(split="val")
    log.info("Validation dataset loaded")

    val_data_loader = init_data_loader(val_dataset)

    num_classes = len(train_dataset.classes)

    log.info("Creating model")
    model = get_tuned_model(cfg.model, num_classes)

    experiment_cfg.model = model.to(experiment_cfg.device)
    log.info(f"Using model: {experiment_cfg.model}")

    experiment_cfg.criterion = nn.BCEWithLogitsLoss()

    train_cfg = TrainingConfig(train_data_loader, experiment_cfg)

    train_cfg.optimizer = Adam(experiment_cfg.model.parameters(), lr=cfg.learning_rate)

    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    train_cfg.lr_scheduler = MultiStepLR(train_cfg.optimizer, milestones=milestones, gamma=cfg.gamma)

    val_cfg = copy.copy(experiment_cfg)
    val_cfg.data_loader = val_data_loader

    # wandb.init(
    #     project="bachelor-thesis-experiment-dev",
    #     config=OmegaConf.to_object(cfg)
    # )

    log.info("Starting training")
    start_time = time.time()
    for _ in trange(cfg.epochs, desc="Training"):
        train_one_epoch(train_cfg)
        train_cfg.lr_scheduler.step()
        evaluate(val_cfg, leave_pbar=False)
        # TODO: Save model and checkpoint

    total_time = time.time() - start_time
    delta = datetime.timedelta(seconds=int(total_time))
    log.info(f"Training done. Took {delta} seconds")

    # Free up memory
    del train_dataset
    del val_dataset

    test_dataset = init_dataset(split="test")

    experiment_cfg.data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

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


def train_one_epoch(cfg: TrainingConfig):
    cfg.model.train()
    # TODO: https://discuss.pytorch.org/t/recording-loss-history-without-i-o/22611
    for _, (images, targets) in tenumerate(cfg.data_loader, desc="Learning",
                                           total=len(cfg.data_loader),
                                           leave=False):
        cfg.optimizer.zero_grad()

        images = images.to(cfg.device)
        predictions = cfg.model(images)

        targets = targets.to(torch.float32).to(cfg.device)
        loss = cfg.criterion(predictions, targets)

        loss.backward()
        cfg.optimizer.step()


def evaluate(cfg: ExperimentConfig, leave_pbar: bool):
    # testing
    cfg.model.eval()
    dataset = cfg.data_loader.dataset
    predictions_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    targets_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    with torch.no_grad():
        for i, (images, targets) in tenumerate(cfg.data_loader, desc="Evaluating", total=len(cfg.data_loader),
                                               leave=leave_pbar):
            images = images.to(cfg.device)
            predictions = cfg.model(images)

            targets = targets.to(torch.float32).to(cfg.device)

            for j in range(min(cfg.data_loader.batch_size, len(predictions))):
                tensor_index = i * cfg.data_loader.batch_size + j
                predictions_t[tensor_index] = predictions[j]
                targets_t[tensor_index] = targets[j]

    return predictions_t, targets_t


if __name__ == '__main__':
    main()
