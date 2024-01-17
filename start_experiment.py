import torch
from utils.utils import plot_training
import torch.nn.functional as F


def compute_accuracy(preds, labels):
    _, pred_labels = torch.max(preds, 1)
    correct = (pred_labels == labels).sum().item()
    accuracy = correct / labels.shape[0]
    return accuracy


def start_training(model, data, epochs, lr=0.001, weight_decay=5e-4, log=True, val=None):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()

            with torch.no_grad():
                test_out = model(data)
                test_loss = F.cross_entropy(test_out[data.test_mask], data.y[data.test_mask])
                test_losses.append(test_loss.item())

            train_acc = compute_accuracy(out[data.train_mask], data.y[data.train_mask])
            test_acc = compute_accuracy(test_out[data.test_mask], data.y[data.test_mask])
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if log:
                print(
                    f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

            train_losses.append(train_loss.item())

            model.train()

    plot_training(train_losses, test_losses, train_accs, test_accs, lr, val, epochs)
