import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
# import wandb
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode, transforms
from tqdm.auto import tqdm, trange

from chest_dataset import ChestVisionDataset


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    composed_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    if cfg.model.startswith("resnet"):
        if cfg.model == "resnet18" or cfg.model == "resnet":
            model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif cfg.model == "resnet50":
            model_ft = resnet50(weights=ResNet50_Weights.DEFAULT)
        n_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(n_features, 14)

    model = model_ft.to(device)

    optim = Adam(model.parameters(), lr=cfg.learning_rate)
    milestones = map(int, [.5 * cfg.epochs, .75 * cfg.epochs])
    lr_scheduler = MultiStepLR(optim, milestones=milestones, gamma=cfg.gamma)

    criterion = nn.BCEWithLogitsLoss()

    print(f"Loading {cfg.dataset} dataset ...")

    npz_file_path = f"/mnt/jbrockma/bachelor-thesis/npz/{cfg.dataset}.npz"

    train_dataset = ChestVisionDataset(npz_file_path, to_rgb=True,
                                       transform=composed_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # wandb.init(
    #     project="bachelor-thesis-experiment-dev",
    #     config=OmegaConf.to_object(cfg)
    # )

    # training
    model.train()
    for _ in trange(cfg.epochs, desc="Training"):
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader), desc="Learning", total=len(train_dataloader), leave=False):
            optim.zero_grad()

            inputs = inputs.to(device)
            outputs = model(inputs)

            labels = labels.to(torch.float32).to(device)
            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

        lr_scheduler.step()

    test_dataset = ChestVisionDataset(npz_file_path, split="test", to_rgb=True,
                                      transform=composed_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    # testing
    model.eval()
    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)

            labels = labels.to(torch.float32).to(device)
            y_true = torch.cat((y_true, labels), 0)
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

    # wandb.finish()


if __name__ == '__main__':
    main()
