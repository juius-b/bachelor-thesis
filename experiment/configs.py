from dataclasses import dataclass
from torch import device, nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Optional, List


@dataclass
class PathsConfig:
    dataset_dir: str
    output_dir: str


@dataclass
class WandbConfig:
    project: str
    add_tags: List[str]


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    wandb: WandbConfig
    model: str
    dataset: str
    epochs: int
    batch_size: int
    iter_size: int
    learning_rate: float
    gamma: float
    checkpoint: Optional[str]


@dataclass(init=False)
class StageConfig:
    device: device
    model: nn.Module
    criterion: nn.Module
    targets_criterion_transform: Callable[[Tensor], Tensor]
    data_loader: DataLoader


@dataclass
class ExtendedStageConfig(StageConfig):
    def __init__(self, data_loader: DataLoader, experiment_cfg: StageConfig):
        for attr_name, attr_value in vars(experiment_cfg).items():
            setattr(self, attr_name, attr_value)
        self.data_loader = data_loader


@dataclass
class TrainingConfig(ExtendedStageConfig):
    optimizer: Optimizer
    iter_size: int

    def __init__(self, data_loader: DataLoader, experiment_cfg: StageConfig):
        super().__init__(data_loader, experiment_cfg)


@dataclass
class EvaluationConfig(ExtendedStageConfig):
    is_multilabel_problem: bool

    def __init__(self, data_loader: DataLoader, experiment_cfg: StageConfig):
        super().__init__(data_loader, experiment_cfg)
