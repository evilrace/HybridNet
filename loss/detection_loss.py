import torch

class det_loss(torch.nn.Module, ):
    def __init__(self, detection_class_num) -> None:
        super().__init__()

        self._delta = 1.0
        self._alpha = 0.256
        self._gamma = 2.0
        self.detection_class_num = detection_class_num

        self._cls_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, y):
        y = y.to('cuda')
        pred_convert = list()
        for p in pred:
            p = p.permute(0,-1,-2,1)
            p = p.reshape(p.size()[0], -1, self.detection_class_num + 1 + 4)
            pred_convert.append(p)
            
        pred_convert = torch.concatenate(pred_convert, dim=1)
        
        pred_cls = pred_convert[:,:,:-4]
        pred_cls = pred_cls.transpose(-1,-2)
        pred_box = pred_convert[:,:,-4:]
        y_cls = y[:,:,0].to(torch.int64)
        y_box = y[:,:,1:]

        cls_loss = self.cls_loss(pred_cls, torch.where(y_cls<0, 0, y_cls))
        box_loss = self.box_loss(pred_box, y_box)

        cls_ignore_mask = torch.where(y_cls >= 0, 1, 0)
        box_ignore_mask = torch.where(y_cls >= 1, 1, 0)
        cls_loss = cls_loss * cls_ignore_mask
        box_loss = box_loss * box_ignore_mask.unsqueeze(-1)

        cls_loss = cls_loss.sum() / cls_ignore_mask.sum()
        box_loss = box_loss.sum() / box_ignore_mask.sum()
        loss = cls_loss + box_loss
        return loss
    
    def cls_loss(self, preds, y):
        loss = self._cls_entropy_loss(preds, y)
        return loss

    def box_loss(self,preds, y):
        squared_difference = torch.pow(preds - y, 2)
        absolute_difference = torch.abs(preds - y)

        loss = torch.where(
            torch.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - self._delta * 0.5,
        )
        return loss