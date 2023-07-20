import copy
import logging
import os
from dataclasses import dataclass
from typing import Callable

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn, device, Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import get_model_weights, WeightsEnum
from torchvision.transforms import transforms
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from tqdm.contrib.logging import logging_redirect_tqdm

from dataset import create_dataset_init, NpzVisionDataset, get_dataset_problem
from tuning import get_tuned_model


@dataclass(init=False)
class ExperimentConfig:
    device: device
    model: nn.Module
    criterion: nn.Module
    targets_criterion_transform: Callable[[Tensor], Tensor]
    data_loader: DataLoader


@dataclass
class ExtendedExperimentConfig(ExperimentConfig):
    def __init__(self, data_loader: DataLoader, experiment_cfg: ExperimentConfig):
        for attr_name, attr_value in vars(experiment_cfg).items():
            setattr(self, attr_name, attr_value)
        self.data_loader = data_loader


@dataclass
class TrainingConfig(ExtendedExperimentConfig):
    optimizer: Optimizer

    def __init__(self, data_loader: DataLoader, experiment_cfg: ExperimentConfig):
        super().__init__(data_loader, experiment_cfg)


@dataclass
class EvaluationConfig(ExtendedExperimentConfig):
    is_multilabel_problem: bool

    def __init__(self, data_loader: DataLoader, experiment_cfg: ExperimentConfig):
        super().__init__(data_loader, experiment_cfg)


log = logging.getLogger("EXPERIMENT")


def create_dataloader_init(batch_size):
    def init(dataset: Dataset, **kwargs):
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True, **kwargs)

    return init


def get_default_model_weights(name: str) -> WeightsEnum:
    weights = get_model_weights(name)
    return getattr(weights, "DEFAULT")


@torch.inference_mode()
def non_multilabel_accuracy_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    predictions = torch.argmax(outputs, dim=1, keepdim=True)
    pred_eq_targets = predictions.eq(targets)
    num_correct = torch.all(pred_eq_targets, dim=1).sum()
    num_total = torch.tensor(targets.shape[0], device=targets.device)
    return (num_correct / num_total).item()


@torch.inference_mode()
def multilabel_accuracy_measure(targets: torch.Tensor, outputs: torch.Tensor, threshold: float = 0.5,
                                exact=True) -> float:
    out_lt_threshold = outputs.lt(threshold)
    predictions = torch.where(out_lt_threshold, 0, 1)
    if exact:
        num_correct = torch.all(predictions.eq(targets), dim=1).sum()
        num_total = torch.tensor(targets.shape[0], device=targets.device)
    else:
        num_correct = predictions.eq(targets).sum()
        num_total = torch.tensor(targets.numel(), device=targets.device)
    return (num_correct / num_total).item()


def binary_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = torch.squeeze(targets).cpu().numpy()
    y_score = outputs[:, 1].cpu().numpy()

    return roc_auc_score(y_true, y_score)


def multiclass_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    with torch.inference_mode():
        n_samples, n_classes = outputs.shape
        y_trues = torch.empty((n_classes, n_samples))
        y_scores = torch.empty((n_classes, n_samples))

        squeezed_targets = torch.squeeze(targets)

        for i in range(n_classes):
            y_trues[i] = torch.where(squeezed_targets.eq(i), 1, 0)
            y_scores[i] = outputs[:, i]

    y_trues, y_scores = y_trues.cpu().numpy(), y_scores.cpu().numpy()

    roc_auc_scores = [roc_auc_score(y_trues[i], y_scores[i]) for i in range(n_classes)]

    return sum(roc_auc_scores) / n_classes


def multilabel_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = targets.cpu().numpy()
    y_score = outputs.cpu().numpy()

    return roc_auc_score(y_true, y_score)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.debug(f"Using config: {cfg}")

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

    problem_type = get_dataset_problem(cfg.dataset.name)

    log.info(f"Recognized classification problem: {problem_type}")

    if problem_type == "multilabel":
        experiment_cfg.criterion = nn.BCEWithLogitsLoss()
        experiment_cfg.targets_criterion_transform = nn.Identity()
    else:
        experiment_cfg.criterion = nn.CrossEntropyLoss()
        experiment_cfg.targets_criterion_transform = lambda ts: torch.squeeze(ts, 1).to(torch.uint8)

    train_cfg = TrainingConfig(train_data_loader, experiment_cfg)
    train_cfg.optimizer = Adam(experiment_cfg.model.parameters(), lr=cfg.learning_rate)

    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    lr_scheduler = MultiStepLR(train_cfg.optimizer, milestones=milestones, gamma=cfg.gamma)

    eval_cfg = EvaluationConfig(val_data_loader, experiment_cfg)
    eval_cfg.is_multilabel_problem = problem_type == "multilabel"

    wandb.init(
        project=f"bachelor-thesis-experiment-{cfg.dataset.name}-dev",
        config=OmegaConf.to_object(cfg)
    )

    best_model = copy.deepcopy(experiment_cfg.model)
    best_auc = 0
    best_epoch = 0

    auc_fn = roc_auc_score
    acc_fn = non_multilabel_accuracy_measure

    match problem_type:
        case "binary":
            auc_fn = binary_auc_measure
        case "multiclass":
            auc_fn = multiclass_auc_measure
        case "multilabel":
            def _acc_fn(_ts, _os):
                return multilabel_accuracy_measure(_ts, _os, exact=False)
            auc_fn = multilabel_auc_measure
            acc_fn = _acc_fn

    output_dir = os.getcwd()

    log.info(f"Starting training with learning rate {lr_scheduler.get_last_lr()[0]:f}")

    with logging_redirect_tqdm():
        for epoch in trange(cfg.epochs, desc="Training"):
            log.debug(f"Starting epoch {epoch + 1}")

            train_one_epoch(train_cfg)
            lr_scheduler.step()

            log.debug(f"Epoch done. Learning rate now: {lr_scheduler.get_last_lr()[0]:f}")
            log.debug("Starting validation")

            targets, outputs, val_loss = evaluate(eval_cfg, leave_pbar=False)
            val_acc = acc_fn(targets, outputs)
            val_auc = auc_fn(targets, outputs)

            log.debug(f"Validation done. AUC@{val_auc:.2f}, ACC@{val_acc:.2f}, LOSS@{val_loss:.2f}")
            wandb.log({"val_auc": val_auc, "val_acc": val_acc, "val_loss": val_loss})

            if best_auc < val_auc:
                best_epoch = epoch
                best_auc = val_auc
                best_model = copy.deepcopy(experiment_cfg.model)
                log.info(f"Current best model in epoch {best_epoch + 1}; AUC@{best_auc:.2f}")

            checkpoint = {
                "model": train_cfg.model.state_dict(),
                "optimizer": train_cfg.optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "cfg": cfg
            }

            torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pth"))

    log.info(f"Training done.")

    # Free up memory
    del train_dataset
    del val_dataset

    state = {
        "model": best_model.state_dict()
    }

    torch.save(state, os.path.join(output_dir, f"model_{best_epoch}@{best_auc:.2f}AUC.pth"))

    test_dataset = init_dataset(split="test")

    eval_cfg.data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    eval_cfg.model = best_model

    targets, outputs, test_loss = evaluate(eval_cfg)

    test_auc = auc_fn(targets, outputs)
    test_acc = acc_fn(targets, outputs)

    log.info(f"Model evaluated: AUC@{test_auc:.2f} & ACC@{test_acc:.2f} (& LOSS@{test_loss:.2f})")
    wandb.log({"test_auc": test_auc, "test_acc": test_acc, "test_loss": test_loss})

    wandb.finish()


def train_one_epoch(cfg: TrainingConfig):
    cfg.model.train()

    loss_sum = torch.tensor(0., device=cfg.device)

    for images, targets in tqdm(cfg.data_loader, desc="Learning", total=len(cfg.data_loader), leave=False):
        cfg.optimizer.zero_grad()

        images, targets = images.to(cfg.device), targets.to(images.dtype).to(cfg.device)

        outputs = cfg.model(images)
        criterion_targets = cfg.targets_criterion_transform(targets)
        loss = cfg.criterion(outputs, criterion_targets)
        loss_sum += loss.detach()

        loss.backward()
        cfg.optimizer.step()

    train_loss = loss_sum / len(cfg.data_loader)
    log.debug(f"Training loss this epoch: {train_loss:.2f}")
    wandb.log({"train_loss": train_loss})


def evaluate(cfg: EvaluationConfig, leave_pbar: bool = True):
    cfg.model.eval()
    dataset: NpzVisionDataset = getattr(cfg.data_loader, "dataset")
    dataset_len = len(dataset)
    targets_t_size = (dataset_len, len(dataset.classes)) if cfg.is_multilabel_problem else (dataset_len, 1)
    targets_t = torch.empty(targets_t_size, device=cfg.device)
    outputs_t = torch.empty((dataset_len, len(dataset.classes)), device=cfg.device)
    loss_sum = torch.tensor(0., device=cfg.device)
    with torch.inference_mode():
        for i, (images, targets) in tenumerate(cfg.data_loader, desc="Evaluating", total=len(cfg.data_loader),
                                               leave=leave_pbar):
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(images.dtype).to(cfg.device, non_blocking=True)

            outputs = cfg.model(images)
            criterion_targets = cfg.targets_criterion_transform(targets)
            loss_sum += cfg.criterion(outputs, criterion_targets)

            tensor_index_start = i * cfg.data_loader.batch_size
            tensor_index_stop = tensor_index_start + min(cfg.data_loader.batch_size, len(outputs))

            targets_t[tensor_index_start:tensor_index_stop] = targets
            outputs_t[tensor_index_start:tensor_index_stop] = outputs

    return targets_t, outputs_t, loss_sum / len(cfg.data_loader)


if __name__ == '__main__':
    main()
