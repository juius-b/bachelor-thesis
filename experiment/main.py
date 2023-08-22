import copy
import hydra
import logging
import os
import pandas as pd
import torch
import wandb
from functools import partial
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.models import get_model_weights
from torchvision.transforms import transforms
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from tqdm.contrib.logging import logging_redirect_tqdm

from configs import ExperimentConfig, StageConfig, TrainingConfig, EvaluationConfig
from dataset import NpzVisionDataset, get_dataset_problem, get_dataset
from measures import non_multilabel_accuracy_measure, multilabel_accuracy_measure, binary_auc_measure, \
    multiclass_auc_measure, multilabel_auc_measure
from tuning import get_tuned_model

log = logging.getLogger("EXPERIMENT")

cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig):
    log.debug(f"Using config: {cfg}")

    stage_config = StageConfig()

    stage_config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log.info(f"Using device: {stage_config.device}")

    weights_enum = get_model_weights(cfg.model)
    default_weights: weights_enum = getattr(weights_enum, "DEFAULT")
    # of type torchvision.transforms._presets.ImageClassification
    preprocess: partial[...] = getattr(default_weights, "transforms")
    size: int = preprocess.keywords["crop_size"]
    model_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Lambda(lambda img: img.convert("RGB")),  # all pre-trained models of torchvision use rgb input
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    npz_file_path = os.path.join(cfg.paths.dataset_dir, f"{cfg.dataset}.npz")

    init_dataset = partial(get_dataset, cfg.dataset, root=npz_file_path, transform=model_transform)

    log.info("Loading training dataset")

    train_dataset = init_dataset()

    log.info("Training dataset loaded")

    init_data_loader = partial(DataLoader, batch_size=cfg.batch_size, pin_memory=True)

    train_data_loader = init_data_loader(train_dataset, shuffle=True)

    log.info("Loading validation dataset")

    val_dataset = init_dataset(split="val")

    log.info("Validation dataset loaded")

    val_data_loader = init_data_loader(val_dataset)

    num_classes = len(train_dataset.classes)

    log.info("Creating model")

    model = get_tuned_model(cfg.model, num_classes)

    stage_config.model = model.to(stage_config.device)

    log.info(f"Using model {cfg.model} of type {stage_config.model.__class__.__name__}")

    problem_type = get_dataset_problem(cfg.dataset)

    log.info(f"Recognized classification problem: {problem_type}")

    if problem_type == "multilabel":
        stage_config.criterion = nn.BCEWithLogitsLoss()
        stage_config.targets_criterion_transform = nn.Identity()
    else:
        stage_config.criterion = nn.CrossEntropyLoss()
        stage_config.targets_criterion_transform = lambda ts: torch.squeeze(ts, 1).to(torch.uint8)

    train_cfg = TrainingConfig(train_data_loader, stage_config)
    train_cfg.optimizer = Adam(stage_config.model.parameters(), lr=cfg.learning_rate)
    train_cfg.iter_size = cfg.iter_size

    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    lr_scheduler = MultiStepLR(train_cfg.optimizer, milestones=milestones, gamma=cfg.gamma)

    eval_cfg = EvaluationConfig(val_data_loader, stage_config)
    eval_cfg.is_multilabel_problem = problem_type == "multilabel"

    best_model = copy.deepcopy(stage_config.model)
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
            auc_fn = multilabel_auc_measure
            acc_fn = partial(multilabel_accuracy_measure, exact=False)

    output_dir = cfg.paths.output_dir if cfg.paths.output_dir else os.getcwd()
    output_dir = Path(output_dir)

    start_epoch = 0

    tags = [cfg.model, cfg.dataset] + cfg.wandb.add_tags

    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        train_cfg.optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        wandb_id = checkpoint["wandb_id"]
        tags.append("resumed")

        log.info("Found checkpoint to resume from")
    else:
        wandb_id = wandb.util.generate_id()

    wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_object(cfg),
        tags=tags,
        resume="allow",
        id=wandb_id
    )

    checkpoint = {
        "cfg": cfg,
        "wandb_id": wandb_id
    }

    log.info(f"Starting training with learning rate {lr_scheduler.get_last_lr()[0]:f}")
    log.info(f"Effective batch size: {cfg.batch_size * cfg.iter_size}")

    with logging_redirect_tqdm():
        for epoch in trange(start_epoch, cfg.epochs, desc="Training"):
            log.debug(f"Starting epoch {epoch}")

            train(train_cfg)
            lr_scheduler.step()

            log.debug(f"Epoch done. Learning rate now: {lr_scheduler.get_last_lr()[0]:f}")
            log.debug("Starting validation")

            targets, outputs, val_loss = evaluate(eval_cfg, leave_pbar=False)
            val_acc = acc_fn(targets, outputs)
            val_auc = auc_fn(targets, outputs)

            pd.DataFrame(outputs.cpu()).to_csv(
                (output_dir / f"{cfg.dataset}-val-{epoch}@{cfg.model}({val_auc:.2f},{val_acc:.2f}).csv"), index=False)

            log.debug(f"Validation done. AUC@{val_auc:.2f}, ACC@{val_acc:.2f}, LOSS@{val_loss:.2f}")
            wandb.log({"val_auc": val_auc, "val_acc": val_acc, "val_loss": val_loss})

            if best_auc < val_auc:
                best_epoch = epoch
                best_auc = val_auc
                best_model = copy.deepcopy(stage_config.model)

                log.info(f"Current best model in epoch {best_epoch}: AUC@{best_auc:.2f}")

            checkpoint["model"] = train_cfg.model.state_dict(),
            checkpoint["optimizer"]: train_cfg.optimizer.state_dict()
            checkpoint["lr_scheduler"]: lr_scheduler.state_dict()
            checkpoint["epoch"]: epoch

            torch.save(checkpoint, (output_dir / f"{cfg.model}-{cfg.dataset}-checkpoint.pth"))

    log.info(f"Training done.")

    # Free up memory
    del train_dataset
    del val_dataset

    state = {
        "model": best_model.state_dict()
    }

    torch.save(state, (output_dir / f"model_{best_epoch}@{best_auc:.2f}AUC.pth"))

    test_dataset = init_dataset(split="test")

    eval_cfg.data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    eval_cfg.model = best_model

    targets, outputs, test_loss = evaluate(eval_cfg)

    test_auc = auc_fn(targets, outputs)
    test_acc = acc_fn(targets, outputs)

    pd.DataFrame(outputs.cpu()).to_csv(
        (output_dir / f"{cfg.dataset}-test@{cfg.model}({test_auc:.2f},{test_acc:.2f}).csv"), index=False)

    log.info(f"Model evaluated: AUC@{test_auc:.2f} & ACC@{test_acc:.2f} (& LOSS@{test_loss:.2f})")
    wandb.log({"test_auc": test_auc, "test_acc": test_acc, "test_loss": test_loss})

    wandb.finish()


def train(cfg: TrainingConfig):
    cfg.model.train()

    loss_sum = torch.tensor(0., device=cfg.device)

    n_batches = 0

    cfg.optimizer.zero_grad()
    for images, targets in tqdm(cfg.data_loader, desc="Learning", total=len(cfg.data_loader), leave=False):
        images, targets = images.to(cfg.device), targets.to(images.dtype).to(cfg.device)

        outputs = cfg.model(images)
        criterion_targets = cfg.targets_criterion_transform(targets)
        loss = cfg.criterion(outputs, criterion_targets)
        loss_sum += loss.detach()

        loss.backward()

        n_batches += 1

        if n_batches % cfg.iter_size == 0 or n_batches == len(cfg.data_loader):
            cfg.optimizer.step()
            cfg.optimizer.zero_grad()

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
