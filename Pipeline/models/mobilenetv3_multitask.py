import torch
import torch.nn as nn
from models.mobilenetv3 import mobilenet_v3_large

class MobileNetV3MultiTask(nn.Module):
    """
    MobileNetV3 com duas cabeças:
    1. Embedding head (para reconhecimento)
    2. Landmark head (loss auxiliar)
    """
    
    def __init__(self, embedding_dim=512, num_landmarks=10):
        super().__init__()
        
        # Backbone MobileNetV3
        base_model = mobilenet_v3_large(embedding_dim=embedding_dim)
        
        # Usa features do MobileNetV3
        self.features = base_model.features
        
        # GDC layer original (para embedding)
        self.gdc = base_model.output_layer
        
        # Cabeça de landmarks (branch auxiliar)
        # Pega features antes do GDC para manter informação espacial
        lastconv_channels = 960  # MobileNetV3-Large antes do GDC
        
        self.landmark_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lastconv_channels, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks)  # 10 valores: (x,y) * 5 landmarks
        )
        
        # Inicialização da cabeça de landmarks
        for m in self.landmark_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_landmarks=True):
        """
        Args:
            x: imagem [B, 3, 112, 112]
            return_landmarks: se True, retorna landmarks preditos
        
        Returns:
            embedding: [B, 512]
            landmarks: [B, 10] (opcional)
        """
        # Features do backbone
        features = self.features(x)  # [B, 960, 7, 7] para MobileNetV3-Large
        
        # Embedding para reconhecimento
        embedding = self.gdc(features)  # [B, 512]
        
        if return_landmarks:
            # Predição de landmarks (usa features antes do GDC)
            landmarks = self.landmark_head(features)  # [B, 10]
            return embedding, landmarks
        else:
            return embedding
    
    def extract_features(self, x):
        """Para inferência: apenas embedding"""
        return self.forward(x, return_landmarks=False)


def mobilenetv3_large_multitask(embedding_dim=512, **kwargs):
    """Factory function"""
    return MobileNetV3MultiTask(embedding_dim=embedding_dim, **kwargs)