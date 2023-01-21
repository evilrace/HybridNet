import torch

class DetLoss(torch.nn.Module):
    def __init__(self, detection_class_num) -> None:
        super().__init__()

        self._delta = 1.0
        self._alpha = 0.256
        self._gamma = 2.0
        self._detection_class_num = detection_class_num
        self._cls_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, y):
        pred_convert = list()
        for p in pred:
            p = p.permute(0,3,2,1)
            p = p.reshape(p.size()[0], -1, self._detection_class_num + 1 + 4)
            pred_convert.append(p)
            
        pred_convert = torch.concatenate(pred_convert, dim=1)
        
        pred_cls = pred_convert[:,:,:-4]
        pred_cls = pred_cls.transpose(-1,-2)
        pred_box = pred_convert[:,:,-4:]
        y_cls = y[:,:,0].to(torch.int64)
        y_box = y[:,:,1:]

        cls_loss = self._cls_loss(pred_cls, torch.where(y_cls<0, 0, y_cls))
        box_loss = self._box_loss(pred_box, y_box)

        cls_positive_mask = torch.where(y_cls > 0, 1, 0)
        cls_negative_mask = torch.where(y_cls == 0, 1, 0)
        box_mask = torch.where(y_cls >= 1, 1, 0)

        cls_positive_loss = cls_loss * cls_positive_mask
        cls_negative_loss = cls_loss * cls_negative_mask
        positive_len = cls_positive_mask.sum()

        negative_len = torch.min(torch.tensor([positive_len*3, cls_negative_mask.sum()]))
        cls_negative_hard = torch.sort(cls_negative_loss, descending=True)[0][:, :negative_len]
        box_loss = box_loss * box_mask.unsqueeze(-1)

        cls_loss = cls_positive_loss.sum() + cls_negative_hard.sum()
        cls_loss = cls_loss / (positive_len + negative_len + 1e-8)
        box_loss = box_loss.sum() / (box_mask.sum() + 1e-8)

        loss = cls_loss + box_loss
        return loss, cls_loss, box_loss
    
    def _cls_loss(self, preds, y):
        loss = self._cls_entropy_loss(preds, y)
        return loss

    def _box_loss(self,preds, y):
        squared_difference = torch.pow(preds - y, 2)
        absolute_difference = torch.abs(preds - y)

        loss = torch.where(
            torch.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - self._delta * 0.5,
        )
        return loss