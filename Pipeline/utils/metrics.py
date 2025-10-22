import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MarginCosineProduct(nn.Module):
    """
    Margin Cosine Product (CosFace)
    
    Implementa a loss function:
    L = -log(exp(s * (cos(theta_yi) - m)) / (exp(s * (cos(theta_yi) - m)) + sum(exp(s * cos(theta_j)))))
    
    onde:
    - s: scale factor
    - m: margin
    - theta_yi: ângulo entre feature e weight da classe correta
    
    Args:
        in_features: dimensão do embedding (512)
        out_features: número de classes (identidades)
        s: scale factor (default: 30.0)
        m: cosine margin (default: 0.40)
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Weight matrix: [out_features, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label):
        """
        Forward pass
        
        Args:
            input: embeddings [B, in_features]
            label: ground truth labels [B]
        
        Returns:
            output: logits com margin [B, out_features]
        """
        # Normalize features and weights
        # input: [B, in_features]
        # weight: [out_features, in_features]
        
        x = F.normalize(input, p=2, dim=1)  # [B, in_features]
        W = F.normalize(self.weight, p=2, dim=1)  # [out_features, in_features]
        
        # Cosine similarity: cos(theta) = x * W^T
        cosine = F.linear(x, W)  # [B, out_features]
        
        # Apply margin to target class
        # phi(theta) = cos(theta) - m
        phi = cosine - self.m
        
        # One-hot encoding do label
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Aplica margin apenas para classe correta
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale
        output *= self.s
        
        return output
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f's={self.s}, '
                f'm={self.m})')


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    
    Implementa:
    cos(theta + m) onde theta é o ângulo entre feature e weight
    
    Args:
        in_features: dimensão do embedding
        out_features: número de classes
        s: scale factor (default: 64.0)
        m: angular margin (default: 0.50)
        easy_margin: se True, usa easy margin
    """
    
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        """Forward pass"""
        # Normalize
        x = F.normalize(input, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine
        cosine = F.linear(x, W)
        
        # Compute cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class AdditiveMarginProduct(nn.Module):
    """
    SphereFace (A-Softmax)
    Implementa margin multiplicativo no ângulo
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=4):
        super(AdditiveMarginProduct, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label):
        """Forward pass"""
        x = F.normalize(input, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(x, W)
        
        # cos(m * theta)
        cos_theta = cosine
        cos_m_theta = self._compute_cos_m_theta(cos_theta, self.m)
        
        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Output
        output = (one_hot * cos_m_theta) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
    
    def _compute_cos_m_theta(self, cos_theta, m):
        """Compute cos(m*theta) usando Chebyshev polynomial"""
        # Simplified version - para m=4
        if m == 4:
            return 8 * torch.pow(cos_theta, 4) - 8 * torch.pow(cos_theta, 2) + 1
        else:
            # Para outros valores de m, retorna aproximação
            return torch.pow(cos_theta, m)


# Test
if __name__ == '__main__':
    print("Testing margin-based losses...")
    
    # Parâmetros
    batch_size = 32
    embedding_dim = 512
    num_classes = 1000
    
    # Dados fake
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test CosFace (MCP)
    print("\n1. Testing CosFace (MarginCosineProduct)...")
    cosface = MarginCosineProduct(embedding_dim, num_classes, s=30.0, m=0.40)
    output = cosface(embeddings, labels)
    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ CosFace OK!")
    
    # Test ArcFace
    print("\n2. Testing ArcFace...")
    arcface = ArcMarginProduct(embedding_dim, num_classes, s=64.0, m=0.50)
    output = arcface(embeddings, labels)
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ ArcFace OK!")
    
    # Test SphereFace
    print("\n3. Testing SphereFace...")
    sphereface = AdditiveMarginProduct(embedding_dim, num_classes, s=30.0, m=4)
    output = sphereface(embeddings, labels)
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ SphereFace OK!")
    
    print("\n✅ utils/metrics.py OK!")