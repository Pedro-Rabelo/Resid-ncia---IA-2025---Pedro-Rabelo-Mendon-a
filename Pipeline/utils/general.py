import os
import random
import numpy as np
import torch
import logging


# Logger global
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

if not LOGGER.handlers:
    LOGGER.addHandler(console_handler)


def setup_seed(seed=42):
    """
    Setup seed para reprodutibilidade
    
    Args:
        seed: seed value (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Para determinismo completo (pode reduzir performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_on_master(state, save_path):
    """
    Salva checkpoint apenas no processo master (para treinamento distribuído)
    
    Args:
        state: state dict para salvar
        save_path: caminho para salvar
    """
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salva checkpoint
    torch.save(state, save_path)
    LOGGER.info(f"Checkpoint saved: {save_path}")


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: predictions [B, num_classes]
        target: ground truth labels [B]
        topk: tuple of top-k values
    
    Returns:
        list of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_checkpoint(model, checkpoint_path, device='cpu', strict=True):
    """
    Carrega checkpoint do modelo
    
    Args:
        model: modelo PyTorch
        checkpoint_path: caminho para checkpoint
        device: device para carregar
        strict: se True, exige match exato das keys
    
    Returns:
        epoch: epoch do checkpoint
        best_acc: melhor acurácia
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    LOGGER.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Carrega state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_lfw_accuracy', 0.0)
    
    LOGGER.info(f"Checkpoint loaded: epoch={epoch}, best_acc={best_acc:.4f}")
    
    return epoch, best_acc


def count_parameters(model):
    """
    Conta número de parâmetros treináveis
    
    Args:
        model: modelo PyTorch
    
    Returns:
        total: número total de parâmetros
        trainable: número de parâmetros treináveis
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def get_lr(optimizer):
    """
    Retorna learning rate atual do optimizer
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        lr: learning rate atual
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    """
    Define learning rate do optimizer
    
    Args:
        optimizer: PyTorch optimizer
        lr: novo learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_lr=0):
    """
    Warmup do learning rate
    
    Args:
        optimizer: PyTorch optimizer
        epoch: epoch atual
        warmup_epochs: número de epochs de warmup
        base_lr: learning rate base
        warmup_lr: learning rate inicial (default: 0)
    """
    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
        set_lr(optimizer, lr)
        return lr
    return get_lr(optimizer)


class ProgressMeter:
    """
    Mostra progresso do treinamento
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        LOGGER.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def create_logger(log_file=None, log_level=logging.INFO):
    """
    Cria logger customizado
    
    Args:
        log_file: arquivo para salvar logs (opcional)
        log_level: nível de logging
    
    Returns:
        logger: logger configurado
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Test
if __name__ == '__main__':
    # Test logger
    LOGGER.info("Testing general utilities")
    
    # Test AverageMeter
    meter = AverageMeter("Loss", ":.4f")
    meter.update(1.5, n=32)
    meter.update(1.3, n=32)
    print(f"Average meter: {meter}")
    
    # Test seed
    setup_seed(42)
    print(f"Random number (should be same): {torch.rand(1).item():.6f}")
    
    setup_seed(42)
    print(f"Random number (should be same): {torch.rand(1).item():.6f}")
    
    print("\n✅ utils/general.py OK!")