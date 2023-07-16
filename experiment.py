import copy
import hydra
import logging
import os
import torch
from dataclasses import dataclass
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
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

from dataset import create_dataset_init, NpzVisionDataset, get_dataset_problem
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


def binary_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = targets.cpu().numpy().flatten()
    y_score = outputs.cpu().numpy().flatten()

    return roc_auc_score(y_true, y_score)


def multiclass_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = targets.cpu().numpy()
    y_score = outputs.cpu().numpy()

    return roc_auc_score(y_true, y_score, multi_class="ovr")


def multilabel_auc_measure(targets: torch.Tensor, outputs: torch.Tensor) -> float:
    y_true = targets.cpu().numpy()
    y_score = outputs.cpu().numpy()

    return roc_auc_score(y_true, y_score)


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

    problem = get_dataset_problem(cfg.dataset.name)

    experiment_cfg.criterion = nn.BCEWithLogitsLoss() if problem == "multilabel" else nn.CrossEntropyLoss()

    train_cfg = TrainingConfig(train_data_loader, experiment_cfg)

    train_cfg.optimizer = Adam(experiment_cfg.model.parameters(), lr=cfg.learning_rate)

    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    lr_scheduler = MultiStepLR(train_cfg.optimizer, milestones=milestones, gamma=cfg.gamma)

    val_cfg = copy.copy(experiment_cfg)
    val_cfg.data_loader = val_data_loader

    # wandb.init(
    #     project="bachelor-thesis-experiment-dev",
    #     config=OmegaConf.to_object(cfg)
    # )

    best_model = copy.deepcopy(experiment_cfg.model)
    best_auc = 0
    best_epoch = 0

    auc_fn = roc_auc_score

    match problem:
        case "binary":
            auc_fn = binary_auc_measure
        case "multiclass":
            auc_fn = multiclass_auc_measure
        case "multilabel":
            auc_fn = multilabel_auc_measure

    output_dir = os.getcwd()

    log.info(f"Starting training with learning rate {lr_scheduler.get_last_lr()[0]:f}")

    with logging_redirect_tqdm():
        for epoch in trange(cfg.epochs, desc="Training"):
            log.info(f"Starting epoch {epoch + 1}")

            train_one_epoch(train_cfg)
            lr_scheduler.step()

            log.info(f"Epoch done. Learning rate now: {lr_scheduler.get_last_lr()[0]:f}")
            log.info("Starting validation")

            targets, outputs, losses = evaluate(val_cfg, leave_pbar=False)
            acc = accuracy_measure(targets, outputs)
            auc = auc_fn(targets, outputs)
            # mean_loss = torch.mean(losses, 0).item()

            log.info(f"Validation done. ACC@{acc:.2f}, AUC@{auc:.2f}")

            if best_auc < auc:
                best_epoch = epoch
                best_auc = auc
                best_model = copy.deepcopy(experiment_cfg.model)

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

    experiment_cfg.data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    targets, outputs, _ = evaluate(experiment_cfg, leave_pbar=True)

    auc = auc_fn(targets, outputs)
    acc = accuracy_measure(targets, outputs)

    print(f"AUC: {auc}", f"Accuracy: {acc}")

    # wandb.finish()


def train_one_epoch(cfg: TrainingConfig):
    cfg.model.train()

    for images, targets in tqdm(cfg.data_loader, desc="Learning", total=len(cfg.data_loader), leave=False):
        cfg.optimizer.zero_grad()

        images, targets = images.to(cfg.device), targets.to(torch.float32).to(cfg.device)

        outputs = cfg.model(images)
        loss = cfg.criterion(outputs, targets)

        loss.backward()
        cfg.optimizer.step()


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
