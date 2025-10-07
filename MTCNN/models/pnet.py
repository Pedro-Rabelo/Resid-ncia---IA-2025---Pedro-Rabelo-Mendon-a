import torch
import torch.nn as nn
import torch.nn.functional as F

class PNet(nn.Module):
    """
    Proposal Network (P-Net) - Primeiro estágio do MTCNN
    
    Arquitetura:
    - Input: 12x12x3
    - Conv1: 10 filtros 3x3, stride=1
    - PReLU
    - MaxPool: 2x2, stride=2
    - Conv2: 16 filtros 3x3, stride=1
    - PReLU
    - Conv3: 32 filtros 3x3, stride=1
    - PReLU
    
    Outputs:
    - Face classification: 2 classes (face/non-face)
    - Bounding box regression: 4 valores (left, top, width, height)
    - Landmark localization: 10 valores (5 pontos x,y)
    """
    
    def __init__(self):
        super(PNet, self).__init__()
        
        # ========== Feature Extraction ==========
        # Conv1: 3 -> 10 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0)
        self.prelu1 = nn.PReLU(10)
        
        # Pool1: 2x2 max pooling, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        # Conv2: 10 -> 16 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=0)
        self.prelu2 = nn.PReLU(16)
        
        # Conv3: 16 -> 32 channels, 3x3 kernel
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.prelu3 = nn.PReLU(32)
        
        # ========== Multi-task Outputs ==========
        # Task 1: Face classification (face/non-face)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        
        # Task 2: Bounding box regression (4 offsets)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        
        # Task 3: Facial landmark localization (10 coordinates)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialização de pesos seguindo o paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W] onde H,W >= 12
        
        Returns:
            cls: Face classification [B, 2, H', W']
            box: Bounding box regression [B, 4, H', W']
            landmark: Landmark localization [B, 10, H', W']
        """
        # Feature extraction
        x = self.conv1(x)      # [B, 10, H-2, W-2]
        x = self.prelu1(x)
        x = self.pool1(x)      # [B, 10, (H-2)/2, (W-2)/2]
        
        x = self.conv2(x)      # [B, 16, H', W']
        x = self.prelu2(x)
        
        x = self.conv3(x)      # [B, 32, H'', W'']
        x = self.prelu3(x)
        
        # Multi-task outputs
        cls = self.conv4_1(x)       # Face classification
        box = self.conv4_2(x)       # Bounding box regression
        landmark = self.conv4_3(x)  # Landmark localization
        
        return cls, box, landmark
    
    def predict(self, x):
        """
        Prediction com softmax e reshape para inferência
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            cls_prob: Face probability [B, H', W']
            box_pred: Box offsets [B, 4, H', W']
            landmark_pred: Landmarks [B, 10, H', W']
        """
        cls, box, landmark = self.forward(x)
        
        # Softmax para probabilidades de classificação
        cls_prob = F.softmax(cls, dim=1)[:, 1, :, :]  # Probability of being a face
        
        return cls_prob, box, landmark


class PNetLoss(nn.Module):
    """
    Multi-task Loss para P-Net
    
    L = α1 * L_cls + α2 * L_box + α3 * L_landmark
    """
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=0.5):
        super(PNetLoss, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.landmark_weight = landmark_weight
        
        # Losses individuais
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.box_loss_fn = nn.MSELoss(reduction='none')
        self.landmark_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, cls_pred, box_pred, landmark_pred, 
                cls_target, box_target, landmark_target, sample_type):
        """
        Calcula loss multi-tarefa com sample type indicator
        
        Args:
            cls_pred: [B, 2, H, W] - Predições de classificação
            box_pred: [B, 4, H, W] - Predições de bounding box
            landmark_pred: [B, 10, H, W] - Predições de landmarks
            cls_target: [B] - Ground truth classes
            box_target: [B, 4] - Ground truth boxes
            landmark_target: [B, 10] - Ground truth landmarks
            sample_type: [B] - Tipo de sample (0=neg, 1=pos, 2=part, 3=landmark)
        
        Returns:
            total_loss: Loss total ponderada
            loss_dict: Dicionário com losses individuais
        """
        batch_size = cls_pred.size(0)
        
        # Reshape predictions para cálculo de loss
        # Para P-Net fully convolutional, pegamos valores centrais
        if cls_pred.dim() == 4:
            # Pega o valor central da saída convolucional
            h, w = cls_pred.size(2), cls_pred.size(3)
            cls_pred = cls_pred[:, :, h//2, w//2]
            box_pred = box_pred[:, :, h//2, w//2]
            landmark_pred = landmark_pred[:, :, h//2, w//2]
        
        # ========== Classification Loss ==========
        # Aplicado para: negative (0), positive (1), part (2)
        cls_mask = (sample_type <= 2)
        cls_loss = torch.tensor(0.0, device=cls_pred.device)
        
        if cls_mask.sum() > 0:
            cls_loss_all = self.cls_loss_fn(cls_pred, cls_target)
            cls_loss = (cls_loss_all * cls_mask.float()).sum() / (cls_mask.sum() + 1e-8)
        
        # ========== Bounding Box Regression Loss ==========
        # Aplicado para: positive (1), part (2)
        box_mask = ((sample_type == 1) | (sample_type == 2))
        box_loss = torch.tensor(0.0, device=box_pred.device)
        
        if box_mask.sum() > 0:
            box_loss_all = self.box_loss_fn(box_pred, box_target).sum(dim=1)
            box_loss = (box_loss_all * box_mask.float()).sum() / (box_mask.sum() + 1e-8)
        
        # ========== Landmark Localization Loss ==========
        # Aplicado apenas para: landmark (3)
        landmark_mask = (sample_type == 3)
        landmark_loss = torch.tensor(0.0, device=landmark_pred.device)
        
        if landmark_mask.sum() > 0:
            landmark_loss_all = self.landmark_loss_fn(landmark_pred, landmark_target).sum(dim=1)
            landmark_loss = (landmark_loss_all * landmark_mask.float()).sum() / (landmark_mask.sum() + 1e-8)
        
        # ========== Total Loss ==========
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


if __name__ == "__main__":
    # Test P-Net
    print("="*60)
    print("TESTE DO P-NET")
    print("="*60)
    
    model = PNet()
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 12, 12)
    cls, box, landmark = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Classification output: {cls.shape}")
    print(f"Box regression output: {box.shape}")
    print(f"Landmark output: {landmark.shape}")
    
    # Test prediction
    cls_prob, box_pred, landmark_pred = model.predict(x)
    print(f"\nPrediction - Face probability: {cls_prob.shape}")
    print(f"Prediction - Box: {box_pred.shape}")
    print(f"Prediction - Landmark: {landmark_pred.shape}")
    
    # Test loss
    loss_fn = PNetLoss()
    cls_target = torch.randint(0, 2, (4,))
    box_target = torch.randn(4, 4)
    landmark_target = torch.randn(4, 10)
    sample_type = torch.tensor([0, 1, 2, 3])  # neg, pos, part, landmark
    
    total_loss, loss_dict = loss_fn(cls, box, landmark, 
                                     cls_target, box_target, landmark_target, sample_type)
    
    print(f"\nLoss test:")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    print("\n✓ P-Net testado com sucesso!")