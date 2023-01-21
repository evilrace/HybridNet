import torch


class SegLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._loss_fn = torch.nn.BCELoss()

    def forward(self, pred, y):
        loss = self._loss_fn(pred, y)
        pred = torch.where(pred > 0.5, 1, 0)
        acc = (pred == y).to(torch.uint8)


        return loss, acc

    