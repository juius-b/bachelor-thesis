import torch
from sklearn.metrics import roc_auc_score


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


@torch.inference_mode()
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
