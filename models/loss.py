import torch
import torch.nn as nn
import torch.nn.functional as F

class FCOSLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta # beta parameter for Smooth L1 Loss
        
    def focal_loss(self, logits, targets):
        """
        Sigmoid Focal Loss.
        Args:
            logits: [N, C]
            targets: [N, C]
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    def forward(self, pred_dict, targets_dict):
        """
        Args:
            pred_dict: output of model containing 'cls_logits', 'reg_preds', 'centerness'
            targets_dict: targets containing 'cls_targets', 'reg_targets', 'centerness_targets', 'is_pos'
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss values
        """
        cls_logits = pred_dict["cls_logits"] # [B, M, C]
        reg_preds = pred_dict["reg_preds"] # [B, M, 4]
        pred_centerness = pred_dict["centerness"] # [B, M, 1]
        
        cls_targets = targets_dict["cls_targets"] # [B, M, C]
        reg_targets = targets_dict["reg_targets"] # [B, M, 4]
        centerness_targets = targets_dict["centerness_targets"] # [B, M, 1]
        is_pos = targets_dict["is_pos"] # [B, M]
        
        batch_size = cls_logits.shape[0]
        num_pos = is_pos.sum().item()
        
        # 1. Classification Loss (Focal Loss)
        # Apply Focal Loss on all locations
        cls_loss = self.focal_loss(cls_logits.reshape(-1, cls_logits.shape[-1]), 
                                   cls_targets.reshape(-1, cls_targets.shape[-1]))
        cls_loss = cls_loss.sum() / max(num_pos, 1.0)
        
        # Initialize default regression and centerness losses to 0
        reg_loss = torch.tensor(0.0, device=cls_logits.device)
        ctr_loss = torch.tensor(0.0, device=cls_logits.device)
        
        if num_pos > 0:
            # Filter positive locations
            pos_reg_preds = reg_preds[is_pos] # [num_pos, 4]
            pos_reg_targets = reg_targets[is_pos] # [num_pos, 4]
            pos_centerness_targets = centerness_targets[is_pos] # [num_pos, 1]
            pos_pred_centerness = pred_centerness[is_pos] # [num_pos, 1]
            
            # 2. Bbox Regression Loss (Smooth L1 Loss)
            # Smooth L1 loss: reduction='none' so we can apply centerness weighting
            pos_reg_loss = F.smooth_l1_loss(pos_reg_preds, pos_reg_targets, beta=self.beta, reduction="none") # [num_pos, 4]
            pos_reg_loss = pos_reg_loss.sum(dim=-1, keepdim=True) # [num_pos, 1]
            
            # FCOS weights regression loss by target centerness to suppress low-quality boxes
            reg_loss = (pos_reg_loss * pos_centerness_targets).sum() / max(pos_centerness_targets.sum(), 1e-6)
            
            # 3. Centerness Loss (Binary Cross Entropy)
            pos_ctr_loss = F.binary_cross_entropy_with_logits(pos_pred_centerness, pos_centerness_targets, reduction="mean")
            ctr_loss = pos_ctr_loss
            
        total_loss = cls_loss + reg_loss + ctr_loss
        
        return total_loss, {
            "loss_cls": cls_loss.item(),
            "loss_reg": reg_loss.item(),
            "loss_ctr": ctr_loss.item(),
            "total_loss": total_loss.item()
        }
