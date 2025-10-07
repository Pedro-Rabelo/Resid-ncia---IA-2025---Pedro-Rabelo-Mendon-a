import torch
import torch.nn as nn

class MTCNNLoss(nn.Module):
    """
    Base class para Multi-task Loss do MTCNN
    
    Combina três losses:
    - Classification: CrossEntropy para face/non-face
    - BBox Regression: MSE para offsets de bounding box
    - Landmark: MSE para coordenadas de landmarks
    
    Sample type indicators (β):
    - 0: negative (apenas classification)
    - 1: positive (classification + bbox)
    - 2: part (classification + bbox)
    - 3: landmark (apenas landmark)
    """
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=0.5):
        """
        Args:
            cls_weight: peso da classification loss (α₁)
            box_weight: peso da bbox regression loss (α₂)
            landmark_weight: peso da landmark loss (α₃)
        """
        super(MTCNNLoss, self).__init__()
        
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.landmark_weight = landmark_weight
        
        # Loss functions individuais
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.box_loss_fn = nn.MSELoss(reduction='none')
        self.landmark_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, cls_pred, box_pred, landmark_pred,
                cls_target, box_target, landmark_target, sample_type):
        """
        Calcula multi-task loss com sample type indicators
        
        Args:
            cls_pred: [B, 2] ou [B, 2, H, W] - classificação
            box_pred: [B, 4] ou [B, 4, H, W] - bbox offsets
            landmark_pred: [B, 10] ou [B, 10, H, W] - landmarks
            cls_target: [B] - ground truth classes (0 ou 1)
            box_target: [B, 4] - ground truth offsets
            landmark_target: [B, 10] - ground truth landmarks
            sample_type: [B] - tipo (0=neg, 1=pos, 2=part, 3=landmark)
        
        Returns:
            total_loss: scalar loss
            loss_dict: dicionário com losses individuais
        """
        device = cls_pred.device
        
        # Para P-Net (saída convolucional), pegar valores centrais
        if cls_pred.dim() == 4:
            h, w = cls_pred.size(2), cls_pred.size(3)
            cls_pred = cls_pred[:, :, h//2, w//2]
            box_pred = box_pred[:, :, h//2, w//2]
            landmark_pred = landmark_pred[:, :, h//2, w//2]
        
        # ========== Classification Loss ==========
        # Aplicado para: negative (0), positive (1), part (2)
        cls_mask = (sample_type <= 2)
        cls_loss = torch.tensor(0.0, device=device)
        
        if cls_mask.sum() > 0:
            cls_loss_all = self.cls_loss_fn(cls_pred, cls_target)
            cls_loss = (cls_loss_all * cls_mask.float()).sum() / (cls_mask.sum() + 1e-8)
        
        # ========== Bounding Box Regression Loss ==========
        # Aplicado para: positive (1), part (2)
        box_mask = ((sample_type == 1) | (sample_type == 2))
        box_loss = torch.tensor(0.0, device=device)
        
        if box_mask.sum() > 0:
            box_loss_all = self.box_loss_fn(box_pred, box_target).sum(dim=1)
            box_loss = (box_loss_all * box_mask.float()).sum() / (box_mask.sum() + 1e-8)
        
        # ========== Landmark Localization Loss ==========
        # Aplicado apenas para: landmark (3)
        landmark_mask = (sample_type == 3)
        landmark_loss = torch.tensor(0.0, device=device)
        
        if landmark_mask.sum() > 0:
            landmark_loss_all = self.landmark_loss_fn(landmark_pred, landmark_target).sum(dim=1)
            landmark_loss = (landmark_loss_all * landmark_mask.float()).sum() / (landmark_mask.sum() + 1e-8)
        
        # ========== Total Weighted Loss ==========
        total_loss = (self.cls_weight * cls_loss +
                     self.box_weight * box_loss +
                     self.landmark_weight * landmark_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'box': box_loss.item(),
            'landmark': landmark_loss.item()
        }
        
        return total_loss, loss_dict


class PNetLoss(MTCNNLoss):
    """Loss específica para P-Net"""
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=0.5):
        super(PNetLoss, self).__init__(cls_weight, box_weight, landmark_weight)


class RNetLoss(MTCNNLoss):
    """Loss específica para R-Net"""
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=0.5):
        super(RNetLoss, self).__init__(cls_weight, box_weight, landmark_weight)


class ONetLoss(MTCNNLoss):
    """
    Loss específica para O-Net
    PESO MAIOR para landmarks (1.0 vs 0.5)
    """
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=1.0):
        super(ONetLoss, self).__init__(cls_weight, box_weight, landmark_weight)


class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desbalanceamento de classes
    Opcional: pode ser usado no lugar de CrossEntropy
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WingLoss(nn.Module):
    """
    Wing Loss para landmark localization
    Mais robusto que MSE para outliers
    
    Referência: Wing Loss for Robust Facial Landmark Localisation with CNNs
    """
    
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 10] predicted landmarks
            target: [B, 10] ground truth landmarks
        """
        delta = (pred - target).abs()
        
        # Wing loss formula
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        return loss.mean()