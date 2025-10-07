import torch
import torch.nn as nn
import torch.nn.functional as F

class RNet(nn.Module):
    """
    Refinement Network (R-Net) - Segundo estágio do MTCNN
    
    Arquitetura:
    - Input: 24x24x3
    - Conv1: 28 filtros 3x3, stride=1
    - PReLU + MaxPool 3x3, stride=2
    - Conv2: 48 filtros 3x3, stride=1
    - PReLU + MaxPool 3x3, stride=2
    - Conv3: 64 filtros 2x2, stride=1
    - PReLU
    - FC1: 128 units
    - PReLU
    
    Outputs (via FC layers):
    - Face classification: 2 classes
    - Bounding box regression: 4 valores
    - Landmark localization: 10 valores
    """
    
    def __init__(self):
        super(RNet, self).__init__()
        
        # ========== Convolutional Layers ==========
        # Conv1: 3 -> 28 channels
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=0)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Conv2: 28 -> 48 channels
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=0)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Conv3: 48 -> 64 channels
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0)
        self.prelu3 = nn.PReLU(64)
        
        # ========== Fully Connected Layers ==========
        # Calcular dimensão após convoluções: 24 -> 22 -> 11 -> 9 -> 4 -> 2 -> 1
        # FC input: 64 * 3 * 3 = 576
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU(128)
        
        # ========== Multi-task Outputs ==========
        # Task 1: Face classification
        self.fc2_1 = nn.Linear(128, 2)
        
        # Task 2: Bounding box regression
        self.fc2_2 = nn.Linear(128, 4)
        
        # Task 3: Facial landmark localization
        self.fc2_3 = nn.Linear(128, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialização de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, 24, 24]
        
        Returns:
            cls: Face classification [B, 2]
            box: Bounding box regression [B, 4]
            landmark: Landmark localization [B, 10]
        """
        # Convolutional layers
        x = self.conv1(x)      # [B, 28, 22, 22]
        x = self.prelu1(x)
        x = self.pool1(x)      # [B, 28, 11, 11]
        
        x = self.conv2(x)      # [B, 48, 9, 9]
        x = self.prelu2(x)
        x = self.pool2(x)      # [B, 48, 4, 4]
        
        x = self.conv3(x)      # [B, 64, 3, 3]
        x = self.prelu3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 576]
        
        # Fully connected
        x = self.fc1(x)        # [B, 128]
        x = self.prelu4(x)
        
        # Multi-task outputs
        cls = self.fc2_1(x)       # [B, 2]
        box = self.fc2_2(x)       # [B, 4]
        landmark = self.fc2_3(x)  # [B, 10]
        
        return cls, box, landmark
    
    def predict(self, x):
        """
        Prediction com softmax
        
        Args:
            x: Input tensor [B, 3, 24, 24]
        
        Returns:
            cls_prob: Face probability [B]
            box_pred: Box offsets [B, 4]
            landmark_pred: Landmarks [B, 10]
        """
        cls, box, landmark = self.forward(x)
        
        # Softmax para probabilidades
        cls_prob = F.softmax(cls, dim=1)[:, 1]  # Probability of being a face
        
        return cls_prob, box, landmark


class RNetLoss(nn.Module):
    """
    Multi-task Loss para R-Net
    Idêntico ao P-Net Loss mas sem considerar outputs convolucionais
    """
    
    def __init__(self, cls_weight=1.0, box_weight=0.5, landmark_weight=0.5):
        super(RNetLoss, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.landmark_weight = landmark_weight
        
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.box_loss_fn = nn.MSELoss(reduction='none')
        self.landmark_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, cls_pred, box_pred, landmark_pred,
                cls_target, box_target, landmark_target, sample_type):
        """
        Calcula loss multi-tarefa
        
        Args:
            cls_pred: [B, 2] - Predições de classificação
            box_pred: [B, 4] - Predições de bounding box
            landmark_pred: [B, 10] - Predições de landmarks
            cls_target: [B] - Ground truth classes
            box_target: [B, 4] - Ground truth boxes
            landmark_target: [B, 10] - Ground truth landmarks
            sample_type: [B] - Tipo de sample (0=neg, 1=pos, 2=part, 3=landmark)
        
        Returns:
            total_loss: Loss total
            loss_dict: Losses individuais
        """
        # Classification Loss (neg, pos, part)
        cls_mask = (sample_type <= 2)
        cls_loss = torch.tensor(0.0, device=cls_pred.device)
        
        if cls_mask.sum() > 0:
            cls_loss_all = self.cls_loss_fn(cls_pred, cls_target)
            cls_loss = (cls_loss_all * cls_mask.float()).sum() / (cls_mask.sum() + 1e-8)
        
        # Box Regression Loss (pos, part)
        box_mask = ((sample_type == 1) | (sample_type == 2))
        box_loss = torch.tensor(0.0, device=box_pred.device)
        
        if box_mask.sum() > 0:
            box_loss_all = self.box_loss_fn(box_pred, box_target).sum(dim=1)
            box_loss = (box_loss_all * box_mask.float()).sum() / (box_mask.sum() + 1e-8)
        
        # Landmark Loss (landmark only)
        landmark_mask = (sample_type == 3)
        landmark_loss = torch.tensor(0.0, device=landmark_pred.device)
        
        if landmark_mask.sum() > 0:
            landmark_loss_all = self.landmark_loss_fn(landmark_pred, landmark_target).sum(dim=1)
            landmark_loss = (landmark_loss_all * landmark_mask.float()).sum() / (landmark_mask.sum() + 1e-8)
        
        # Total Loss
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
    print("="*60)
    print("TESTE DO R-NET")
    print("="*60)
    
    model = RNet()
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(8, 3, 24, 24)
    cls, box, landmark = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Classification output: {cls.shape}")
    print(f"Box regression output: {box.shape}")
    print(f"Landmark output: {landmark.shape}")
    
    # Test prediction
    cls_prob, box_pred, landmark_pred = model.predict(x)
    print(f"\nPrediction - Face probability: {cls_prob.shape}")
    
    # Test loss
    loss_fn = RNetLoss()
    cls_target = torch.randint(0, 2, (8,))
    box_target = torch.randn(8, 4)
    landmark_target = torch.randn(8, 10)
    sample_type = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    
    total_loss, loss_dict = loss_fn(cls, box, landmark,
                                     cls_target, box_target, landmark_target, sample_type)
    
    print(f"\nLoss test:")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    print("\n✓ R-Net testado com sucesso!")