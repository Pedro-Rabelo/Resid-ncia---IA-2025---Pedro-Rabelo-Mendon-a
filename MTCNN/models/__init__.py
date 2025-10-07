__version__ = '1.0.0'

try:
    from .pnet import PNet, PNetLoss
except ImportError:
    PNet = None
    PNetLoss = None

try:
    from .rnet import RNet, RNetLoss
except ImportError:
    RNet = None
    RNetLoss = None

try:
    from .onet import ONet, ONetLoss
except ImportError:
    ONet = None
    ONetLoss = None

__all__ = [
    'PNet', 'PNetLoss',
    'RNet', 'RNetLoss',
    'ONet', 'ONetLoss',
]


def get_model_info():
    """Retorna informações sobre os modelos disponíveis"""
    import torch
    
    info = {}
    
    if PNet is not None:
        pnet = PNet()
        info['PNet'] = {
            'input_size': (12, 12),
            'parameters': sum(p.numel() for p in pnet.parameters()),
            'trainable': sum(p.numel() for p in pnet.parameters() if p.requires_grad)
        }
    
    if RNet is not None:
        rnet = RNet()
        info['RNet'] = {
            'input_size': (24, 24),
            'parameters': sum(p.numel() for p in rnet.parameters()),
            'trainable': sum(p.numel() for p in rnet.parameters() if p.requires_grad)
        }
    
    if ONet is not None:
        onet = ONet()
        info['ONet'] = {
            'input_size': (48, 48),
            'parameters': sum(p.numel() for p in onet.parameters()),
            'trainable': sum(p.numel() for p in onet.parameters() if p.requires_grad)
        }
    
    return info