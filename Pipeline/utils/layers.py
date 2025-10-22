import torch
import torch.nn as nn
from typing import Callable, Optional


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Sequential):
    """
    Convolution + BatchNorm + Activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ]
        
        if activation_layer is not None:
            # PReLU não aceita 'inplace', outros sim
            if activation_layer == nn.PReLU:
                layers.append(activation_layer())
            else:
                params = {} if inplace is None else {"inplace": inplace}
                layers.append(activation_layer(**params))
        
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block
    """
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = scale_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class GDC(nn.Module):
    """
    Global Depthwise Convolution (GDC)
    Versão adaptativa que funciona com qualquer tamanho de feature map
    """
    def __init__(self, in_channels: int, embedding_dim: int):
        super(GDC, self).__init__()
        
        # AdaptiveAvgPool2d funciona com qualquer tamanho de entrada
        # Reduz qualquer HxW para 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection para embedding dimension
        self.conv_proj = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.bn_proj = nn.BatchNorm2d(embedding_dim)
        
        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: feature map [B, C, H, W] onde H e W podem ser qualquer tamanho
        
        Returns:
            embedding: [B, embedding_dim]
        """
        # Pooling global adaptativo
        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.pool(x)
        
        # Projeção para embedding dimension
        # [B, C, 1, 1] -> [B, embedding_dim, 1, 1]
        x = self.conv_proj(x)
        x = self.bn_proj(x)
        
        # Flatten
        # [B, embedding_dim, 1, 1] -> [B, embedding_dim]
        x = self.flatten(x)
        
        return x


# Test
if __name__ == '__main__':
    print("Testing layers...")
    
    # Test _make_divisible
    print(f"\n1. _make_divisible(17, 8) = {_make_divisible(17, 8)}")
    print(f"   _make_divisible(23, 8) = {_make_divisible(23, 8)}")
    
    # Test Conv2dNormActivation
    print("\n2. Testing Conv2dNormActivation...")
    conv = Conv2dNormActivation(3, 16, kernel_size=3, stride=2)
    x = torch.randn(2, 3, 224, 224)
    y = conv(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print("   ✅ Conv2dNormActivation OK!")
    
    # Test SqueezeExcitation
    print("\n3. Testing SqueezeExcitation...")
    se = SqueezeExcitation(64, 16)
    x = torch.randn(2, 64, 28, 28)
    y = se(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print("   ✅ SqueezeExcitation OK!")
    
    # Test GDC
    print("\n4. Testing GDC...")
    gdc = GDC(960, 512)
    
    # Teste com diferentes tamanhos
    test_sizes = [(7, 7), (4, 4), (3, 3), (1, 1)]
    for h, w in test_sizes:
        x = torch.randn(2, 960, h, w)
        y = gdc(x)
        print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    print("   ✅ GDC OK! (funciona com qualquer tamanho)")
    
    print("\n✅ utils/layers.py OK!")