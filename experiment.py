import copy
import datetime
import logging
import os
import time
from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch import nn, device
from torch.nn.modules import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import get_model_weights, WeightsEnum
from torchvision.transforms import transforms
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from tqdm.contrib.logging import logging_redirect_tqdm

from dataset import create_dataset_init, NpzVisionDataset, is_multiclass, is_multilabel
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


def accuracy_measure(targets: torch.Tensor, outputs: torch.Tensor, threshold: float = 0.5) -> float:
    with torch.inference_mode():
        predictions = torch.where(outputs.lt(threshold), 0, 1)
        num_correct = torch.all(predictions.eq(targets), dim=1).sum()
        num_total = torch.tensor(targets.shape[0], device=targets.device)
        return (num_correct / num_total).item()


def multiclass_accuracy_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    pass


def auroc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = targets.cpu().numpy()
    y_score = outputs.cpu().numpy()

    return roc_auc_score(y_true, y_score)


def multiclass_auroc_measure():
    pass


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
    log.info(f"Using model {cfg.model} of type {experiment_cfg.model.__class__.__name__}")

    experiment_cfg.criterion = nn.BCEWithLogitsLoss() if is_multilabel(cfg.dataset) else nn.CrossEntropyLoss()

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

    best_model = copy.deepcopy(experiment_cfg.model)
    best_auroc = 0
    best_epoch = 0

    accuracy_fn = multiclass_accuracy_measure if is_multiclass(cfg.dataset) else accuracy_measure
    auroc_fn = multiclass_auroc_measure if is_multiclass(cfg.dataset) else auroc_measure

    log.info(f"Starting training with learning rate {train_cfg.lr_scheduler.get_last_lr()}")
    start_time = time.time()

    with logging_redirect_tqdm():
        for epoch in trange(cfg.epochs, desc="Training"):
            log.info(f"Starting epoch {epoch + 1}")

            train_one_epoch(train_cfg)
            train_cfg.lr_scheduler.step()

            log.info(f"Epoch done. Learning rate now: {train_cfg.lr_scheduler.get_last_lr()}")
            log.info("Starting validation")

            targets, outputs, losses = evaluate(val_cfg, leave_pbar=False)
            accuracy = accuracy_fn(targets, outputs)
            auroc = auroc_fn(targets, outputs)
            # mean_loss = torch.mean(losses, 0).item()

            log.info(f"Validation done. ACC@{accuracy:.2f}, AUC@{auroc:.2f}")

            if best_auroc < auroc:
                best_epoch = epoch
                best_auroc = auroc
                best_model = copy.deepcopy(experiment_cfg.model)
            # TODO: Save model and checkpoint

    total_time = time.time() - start_time
    delta = datetime.timedelta(seconds=int(total_time))
    log.info(f"Training done. Took {delta} seconds")

    # Free up memory
    del train_dataset
    del val_dataset

    state = {
        "model": best_model.state_dict()
    }

    torch.save(state, cfg.output_dir)

    test_dataset = init_dataset(split="test")

    experiment_cfg.data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    y_pred, y_true = evaluate(experiment_cfg, leave_pbar=True)

    y_pred = y_pred.detach().cpu().numpy().squeeze()
    y_true = y_true.detach().cpu().numpy().squeeze()

    auc = 0
    for i in range(y_true.shape[1]):
        label_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc += label_auc
    auc /= y_pred.shape[1]

    y_thresh = y_pred > .5
    accuracy = 0
    for i in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, i], y_thresh[:, i])
        accuracy += label_acc
    accuracy /= y_true.shape[1]

    print(f"AUC: {auc}", f"Accuracy: {accuracy}")

    # wandb.finish()


def train_one_epoch(cfg: TrainingConfig):
    cfg.model.train()

    loss: torch.Tensor = torch.Tensor([0])

    for images, targets in tqdm(cfg.data_loader, desc="Learning", total=len(cfg.data_loader), leave=False):
        cfg.optimizer.zero_grad()

        images, targets = images.to(cfg.device), targets.to(torch.float32).to(cfg.device)

        outputs = cfg.model(images)
        loss = cfg.criterion(outputs, targets)

        loss.backward()
        cfg.optimizer.step()

    log.info(f"Last loss of epoch: {loss.item()}")


def evaluate(cfg: ExperimentConfig, leave_pbar: bool):
    cfg.model.eval()
    dataset: NpzVisionDataset = getattr(cfg.data_loader, "dataset")
    targets_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    outputs_t = torch.empty((len(dataset), len(dataset.classes)), device=cfg.device)
    losses = torch.empty((len(dataset), 1))
    with torch.inference_mode():
        for i, (images, targets) in tenumerate(cfg.data_loader, desc="Evaluating", total=len(cfg.data_loader),
                                               leave=leave_pbar):
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(torch.float32).to(cfg.device, non_blocking=True)

            outputs = cfg.model(images)
            loss = cfg.criterion(outputs, targets)

            for j in range(min(cfg.data_loader.batch_size, len(outputs))):
                tensor_index = i * cfg.data_loader.batch_size + j
                targets_t[tensor_index] = targets[j]
                outputs_t[tensor_index] = outputs[j]
            losses[i] = loss.detach()

    return targets_t, outputs_t, losses


if __name__ == '__main__':
    main()
