import torch



def iou(preds, targets, threshold=.5):
    B, _, H, W = preds.shape
    preds = preds.permute(0, 2, 3, 1).reshape(-1, 4)
    targets = targets.permute(0, 2, 3, 1).reshape(-1, 4)

    pred_x1 = preds[:, 0] - preds[:, 2] / 2
    pred_y1 = preds[:, 1] - preds[:, 3] / 2
    pred_x2 = preds[:, 0] + preds[:, 2] / 2
    pred_y2 = preds[:, 1] + preds[:, 3] / 2

    targ_x1 = targets[:, 0] - targets[:, 2] / 2
    targ_y1 = targets[:, 1] - targets[:, 3] / 2
    targ_x2 = targets[:, 0] + targets[:, 2] / 2
    targ_y2 = targets[:, 1] + targets[:, 3] / 2

    inter_x1 = torch.max(pred_x1, targ_x1)
    inter_y1 = torch.max(pred_y1, targ_y1)
    inter_x2 = torch.min(pred_x2, targ_x2)
    inter_y2 = torch.min(pred_y2, targ_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    targ_area = (targ_x2 - targ_x1).clamp(0) * (targ_y2 - targ_y1).clamp(0)

    union_area = pred_area + targ_area - inter_area + 1e-6
    iou = inter_area / union_area
    
    return (iou > threshold).float().mean().item()

