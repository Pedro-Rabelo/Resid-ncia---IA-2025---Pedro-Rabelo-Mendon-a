import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Loss combinada para multi-task learning:
    1. Classification loss (cross-entropy via MCP)
    2. Landmark regression loss (auxiliar)
    """
    
    def __init__(self, landmark_weight=0.5):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.classification_loss = nn.CrossEntropyLoss()
        self.landmark_loss = nn.SmoothL1Loss()  # Menos sensível a outliers
        
    def forward(self, cls_output, landmarks_pred, targets, landmarks_gt):
        """
        Args:
            cls_output: saída do MCP [B, num_classes]
            landmarks_pred: landmarks preditos [B, 10]
            targets: labels [B]
            landmarks_gt: landmarks ground truth [B, 10]
        
        Returns:
            total_loss: loss combinada
            loss_dict: dicionário com losses individuais
        """
        # Loss de classificação
        L_cls = self.classification_loss(cls_output, targets)
        
        # Loss de landmarks (regressão)
        L_landmark = self.landmark_loss(landmarks_pred, landmarks_gt)
        
        # Loss total ponderada
        total_loss = L_cls + self.landmark_weight * L_landmark
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': L_cls.item(),
            'landmark': L_landmark.item()
        }
        
        return total_loss, loss_dict


class WingLoss(nn.Module):
    """
    Wing Loss para landmarks (melhor que L1/L2)
    Referência: Wing Loss for Robust Facial Landmark Localisation (CVPR 2018)
    """
    
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 10]
            target: [B, 10]
        """
        delta = (target - pred).abs()
        
        # Wing loss formulation
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        return loss.mean()


class MultiTaskLossAdvanced(nn.Module):
    """
    Loss avançada com Wing Loss e pesos adaptativos
    """
    
    def __init__(self, landmark_weight=0.5, use_wing_loss=True):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.classification_loss = nn.CrossEntropyLoss()
        
        if use_wing_loss:
            self.landmark_loss = WingLoss(omega=10, epsilon=2)
        else:
            self.landmark_loss = nn.SmoothL1Loss()
    
    def forward(self, cls_output, landmarks_pred, targets, landmarks_gt):
        # Loss de classificação
        L_cls = self.classification_loss(cls_output, targets)
        
        # Loss de landmarks
        L_landmark = self.landmark_loss(landmarks_pred, landmarks_gt)
        
        # Loss total
        total_loss = L_cls + self.landmark_weight * L_landmark
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': L_cls.item(),
            'landmark': L_landmark.item()
        }
        
        return total_loss, loss_dict