import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F


def evaluation(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (features, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)

            if isinstance(output, torch.distributed.rpc.api.RRef):
                output = output.local_value()

            _, predicts = torch.max(output, 1)
            total += targets.size(0)
            correct += predicts.eq(targets).sum().item()

    accuracy = round(correct / total * 100., 4)
    return accuracy


def epoch_loss(model, data_loader, device):
    model.eval()
    epoch_loss = 0
    total = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            output = model(features)

            if isinstance(output, torch.distributed.rpc.api.RRef):
                output = output.local_value()

            total += targets.size(0)
            loss = F.cross_entropy(output, targets, reduction='sum')
            epoch_loss += loss

    return epoch_loss / total
