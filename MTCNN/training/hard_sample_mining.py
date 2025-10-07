import torch
import numpy as np

class OnlineHardSampleMining:
    """
    Online Hard Sample Mining Strategy
    
    Implementa a estratégia do paper:
    1. No forward pass, calcular loss para TODAS as amostras
    2. Ordenar losses em ordem decrescente
    3. Selecionar top K% como hard samples (default: 70%)
    4. No backward pass, calcular gradientes APENAS para hard samples
    
    Benefícios:
    - Foco automático em samples difíceis
    - Não requer seleção manual
    - Adaptativo ao processo de treinamento
    - Convergência mais rápida
    """
    
    def __init__(self, ratio=0.7, min_samples=16):
        """
        Args:
            ratio: proporção de samples considerados "hard" (0.7 = 70%)
            min_samples: número mínimo de samples a manter
        """
        self.ratio = ratio
        self.min_samples = min_samples
        
    def filter_hard_samples(self, loss_all, sample_type):
        """
        Filtra hard samples baseado nas losses
        
        Args:
            loss_all: tensor [B] com loss individual de cada sample
            sample_type: tensor [B] com tipo de cada sample
        
        Returns:
            hard_mask: tensor booleano [B] indicando hard samples
            stats: dict com estatísticas
        """
        batch_size = loss_all.size(0)
        
        # Separar por tipo de sample
        cls_mask = (sample_type <= 2)  # classification samples (neg, pos, part)
        
        if cls_mask.sum() == 0:
            # Se não há samples de classificação, retornar todos
            return torch.ones_like(sample_type, dtype=torch.bool), {}
        
        # Calcular número de hard samples
        num_cls_samples = cls_mask.sum().item()
        num_hard = max(int(num_cls_samples * self.ratio), self.min_samples)
        num_hard = min(num_hard, num_cls_samples)
        
        # Ordenar losses dos samples de classificação
        cls_losses = loss_all[cls_mask]
        cls_indices = torch.where(cls_mask)[0]
        
        # Top K losses (hard samples)
        _, topk_indices = torch.topk(cls_losses, num_hard, largest=True)
        hard_cls_indices = cls_indices[topk_indices]
        
        # Criar máscara de hard samples
        hard_mask = torch.zeros_like(sample_type, dtype=torch.bool)
        hard_mask[hard_cls_indices] = True
        
        # Incluir TODOS os samples de landmark (sempre são importantes)
        landmark_mask = (sample_type == 3)
        hard_mask = hard_mask | landmark_mask
        
        # Estatísticas
        stats = {
            'total_samples': batch_size,
            'cls_samples': num_cls_samples,
            'hard_samples': num_hard,
            'landmark_samples': landmark_mask.sum().item(),
            'kept_samples': hard_mask.sum().item(),
            'hard_ratio': num_hard / num_cls_samples if num_cls_samples > 0 else 0
        }
        
        return hard_mask, stats
    
    def compute_loss_with_mining(self, loss_fn, cls_pred, box_pred, landmark_pred,
                                  cls_target, box_target, landmark_target, sample_type):
        """
        Calcula loss com hard sample mining
        
        Args:
            loss_fn: função de loss
            cls_pred: [B, 2] predições de classificação
            box_pred: [B, 4] predições de bbox
            landmark_pred: [B, 10] predições de landmarks
            cls_target: [B] ground truth classes
            box_target: [B, 4] ground truth boxes
            landmark_target: [B, 10] ground truth landmarks
            sample_type: [B] tipos de sample
        
        Returns:
            total_loss: loss total (apenas hard samples)
            loss_dict: dicionário com losses e estatísticas
            hard_mask: máscara de hard samples
        """
        device = cls_pred.device
        batch_size = cls_pred.size(0)
        
        # ========== CALCULAR LOSSES INDIVIDUAIS (SEM REDUÇÃO) ==========
        
        # Classification loss
        cls_mask = (sample_type <= 2)
        cls_loss_all = torch.zeros(batch_size, device=device)
        
        if cls_mask.sum() > 0:
            cls_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            cls_loss_all[cls_mask] = cls_loss_fn(cls_pred[cls_mask], cls_target[cls_mask])
        
        # Box regression loss
        box_mask = ((sample_type == 1) | (sample_type == 2))
        box_loss_all = torch.zeros(batch_size, device=device)
        
        if box_mask.sum() > 0:
            box_loss_fn = torch.nn.MSELoss(reduction='none')
            box_loss_all[box_mask] = box_loss_fn(
                box_pred[box_mask], box_target[box_mask]
            ).sum(dim=1)
        
        # Landmark loss
        landmark_mask = (sample_type == 3)
        landmark_loss_all = torch.zeros(batch_size, device=device)
        
        if landmark_mask.sum() > 0:
            landmark_loss_fn = torch.nn.MSELoss(reduction='none')
            landmark_loss_all[landmark_mask] = landmark_loss_fn(
                landmark_pred[landmark_mask], landmark_target[landmark_mask]
            ).sum(dim=1)
        
        # Loss total por sample (para ordenação)
        loss_per_sample = cls_loss_all + box_loss_all + landmark_loss_all
        
        # ========== HARD SAMPLE MINING ==========
        hard_mask, mining_stats = self.filter_hard_samples(loss_per_sample, sample_type)
        
        # ========== CALCULAR LOSS FINAL (APENAS HARD SAMPLES) ==========
        
        # Classification
        hard_cls_mask = cls_mask & hard_mask
        cls_loss = torch.tensor(0.0, device=device)
        if hard_cls_mask.sum() > 0:
            cls_loss = cls_loss_all[hard_cls_mask].mean()
        
        # Box regression
        hard_box_mask = box_mask & hard_mask
        box_loss = torch.tensor(0.0, device=device)
        if hard_box_mask.sum() > 0:
            box_loss = box_loss_all[hard_box_mask].mean()
        
        # Landmarks (sempre incluídos)
        landmark_loss = torch.tensor(0.0, device=device)
        if landmark_mask.sum() > 0:
            landmark_loss = landmark_loss_all[landmark_mask].mean()
        
        # Loss total ponderada
        total_loss = (loss_fn.cls_weight * cls_loss +
                     loss_fn.box_weight * box_loss +
                     loss_fn.landmark_weight * landmark_loss)
        
        # Estatísticas detalhadas
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'box': box_loss.item(),
            'landmark': landmark_loss.item(),
            'mining_stats': mining_stats
        }
        
        return total_loss, loss_dict, hard_mask


class AdaptiveHardSampleMining(OnlineHardSampleMining):
    """
    Versão adaptativa do hard sample mining
    
    Ajusta o ratio dinamicamente baseado na época de treinamento:
    - Início: ratio baixo (mais samples)
    - Meio: ratio médio
    - Final: ratio alto (focar nos mais difíceis)
    """
    
    def __init__(self, initial_ratio=0.5, final_ratio=0.8, min_samples=16):
        super().__init__(ratio=initial_ratio, min_samples=min_samples)
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.current_epoch = 0
        self.total_epochs = 1
    
    def update_ratio(self, epoch, total_epochs):
        """
        Atualiza o ratio baseado na época atual
        
        Args:
            epoch: época atual
            total_epochs: total de épocas
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        # Linear annealing
        progress = epoch / max(1, total_epochs - 1)
        self.ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
        
        return self.ratio


if __name__ == "__main__":
    print("="*60)
    print("TESTE DE HARD SAMPLE MINING")
    print("="*60)
    
    # Simular batch
    batch_size = 128
    device = torch.device('cpu')
    
    # Predições simuladas
    cls_pred = torch.randn(batch_size, 2, device=device)
    box_pred = torch.randn(batch_size, 4, device=device)
    landmark_pred = torch.randn(batch_size, 10, device=device)
    
    # Targets simulados
    cls_target = torch.randint(0, 2, (batch_size,), device=device)
    box_target = torch.randn(batch_size, 4, device=device)
    landmark_target = torch.randn(batch_size, 10, device=device)
    
    # Sample types: 50 neg, 30 pos, 20 part, 28 landmark
    sample_type = torch.cat([
        torch.zeros(50, dtype=torch.long),  # negative
        torch.ones(30, dtype=torch.long),   # positive
        torch.full((20,), 2, dtype=torch.long),  # part
        torch.full((28,), 3, dtype=torch.long)   # landmark
    ])
    sample_type = sample_type[torch.randperm(batch_size)]
    
    # Test online mining
    print("\n[Test 1] Online Hard Sample Mining")
    mining = OnlineHardSampleMining(ratio=0.7)
    
    # Simular losses
    loss_all = torch.rand(batch_size, device=device)
    hard_mask, stats = mining.filter_hard_samples(loss_all, sample_type)
    
    print(f"\nEstatísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nHard samples selecionados: {hard_mask.sum().item()}/{batch_size}")
    
    # Test adaptive mining
    print("\n[Test 2] Adaptive Hard Sample Mining")
    adaptive_mining = AdaptiveHardSampleMining(
        initial_ratio=0.5,
        final_ratio=0.8
    )
    
    print("\nProgresso do ratio através das épocas:")
    for epoch in range(0, 31, 5):
        ratio = adaptive_mining.update_ratio(epoch, total_epochs=30)
        print(f"  Epoch {epoch:2d}: ratio = {ratio:.3f}")
    
    print("\n✓ Hard sample mining testado com sucesso!")
    print("\nBenefícios do Hard Sample Mining:")
    print("  • Foco automático em samples difíceis")
    print("  • Convergência mais rápida")
    print("  • Não requer seleção manual")
    print("  • Adaptativo ao treinamento")