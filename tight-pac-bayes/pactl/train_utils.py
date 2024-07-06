from tqdm.auto import tqdm
import torch

from .distributed import DistributedValue


@torch.no_grad()
def eval_model(model, data_loader, criterion=None, device_id=None, distributed=False, audit=False):
    model.eval()

    losses = []
    N = len(data_loader.dataset)

    nll = torch.tensor(0.).to(device_id)
    N_acc = 0
    if distributed:
        nll = DistributedValue(nll)
        N_acc = DistributedValue(N_acc)

    for X, Y in tqdm(data_loader, leave=False):
        X, Y = X.to(device_id), Y.to(device_id)

        logits = model(X)

        if criterion is not None:
            loss = criterion(logits, Y) * Y.size(0)
            nll += loss
            if audit:
                losses.append(loss)  # fixme: does O(1) need per-sample loss?
        N_acc += (logits.argmax(dim=-1) == Y).sum()

    if distributed:
        nll = nll.resolve()
        N_acc = N_acc.resolve()

    metrics = {'nll': nll.item(), 'acc': N_acc.item() / N, 'avg_nll': nll.item() / N}

    return metrics, losses
