import torch

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.float() if model.task_type == 'regression' else data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    total_error = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if model.task_type == 'regression':
                total_error += F.l1_loss(out, data.y, reduction='sum').item()
            else:
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
    if model.task_type == 'regression':
        return total_error / len(loader.dataset)  # MAE
    else:
        return correct / len(loader.dataset)      # accuracy