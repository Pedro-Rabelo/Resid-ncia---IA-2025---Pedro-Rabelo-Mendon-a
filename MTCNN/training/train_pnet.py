import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pnet import PNet
from models.losses import PNetLoss
from utils.data_utils import MTCNNDataset
from training.hard_sample_mining import OnlineHardSampleMining
from config import Config


def train_pnet():
    """
    Treina P-Net com Online Hard Sample Mining
    """
    print("\n" + "="*70)
    print("TREINAMENTO DO P-NET")
    print("="*70)
    
    # ==================== SETUP ====================
    device = Config.DEVICE
    print(f"\n[Setup] Device: {device}")
    
    # Dataset
    train_anno_file = os.path.join(Config.PROCESSED_DATA_DIR, 'pnet', 'pnet_train.txt')
    
    if not os.path.exists(train_anno_file):
        print(f"\n❌ Erro: {train_anno_file} não encontrado")
        print("Execute: python main.py --mode prepare --stage pnet")
        return
    
    print(f"\n[Dataset] Carregando dados...")
    train_dataset = MTCNNDataset(
        annotation_file=train_anno_file,
        input_size=Config.PNET_INPUT_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.PNET_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    
    print(f"  Samples: {len(train_dataset):,}")
    print(f"  Batches: {len(train_loader):,}")
    print(f"  Batch size: {Config.PNET_BATCH_SIZE}")
    
    # Model
    print(f"\n[Model] Inicializando P-Net...")
    model = PNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function
    loss_fn = PNetLoss(
        cls_weight=Config.PNET_CLS_WEIGHT,
        box_weight=Config.PNET_BOX_WEIGHT,
        landmark_weight=Config.PNET_LANDMARK_WEIGHT
    )
    
    print(f"\n[Loss] Weights: cls={Config.PNET_CLS_WEIGHT}, "
          f"box={Config.PNET_BOX_WEIGHT}, lmk={Config.PNET_LANDMARK_WEIGHT}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.PNET_LR)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20],
        gamma=0.1
    )
    
    # Hard sample mining
    if Config.ONLINE_MINING:
        mining = OnlineHardSampleMining(ratio=Config.HARD_SAMPLE_RATIO)
        print(f"\n[Mining] Online Hard Sample Mining ATIVADO")
        print(f"  Ratio: {Config.HARD_SAMPLE_RATIO}")
    else:
        mining = None
        print(f"\n[Mining] DESATIVADO")
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(Config.TENSORBOARD_DIR, 'pnet'))
    
    # ==================== TRAINING LOOP ====================
    print(f"\n[Training] Iniciando...")
    print(f"  Epochs: {Config.PNET_EPOCHS}")
    print(f"  Learning rate: {Config.PNET_LR}")
    print("-"*70)
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(Config.PNET_EPOCHS):
        model.train()
        
        # Métricas da época
        epoch_losses = {
            'total': [],
            'cls': [],
            'box': [],
            'landmark': []
        }
        
        mining_stats_epoch = {
            'hard_ratio': [],
            'kept_samples': []
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.PNET_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move para device
            images = batch['image'].to(device)
            cls_target = batch['cls_target'].to(device)
            box_target = batch['box_target'].to(device)
            landmark_target = batch['landmark_target'].to(device)
            sample_type = batch['sample_type'].to(device)
            
            # Forward pass
            cls_pred, box_pred, landmark_pred = model(images)
            
            # Calcular loss com hard sample mining
            if mining is not None:
                total_loss, loss_dict, hard_mask = mining.compute_loss_with_mining(
                    loss_fn, cls_pred, box_pred, landmark_pred,
                    cls_target, box_target, landmark_target, sample_type
                )
                
                # Estatísticas de mining
                mining_stats = loss_dict.get('mining_stats', {})
                if mining_stats:
                    mining_stats_epoch['hard_ratio'].append(mining_stats.get('hard_ratio', 0))
                    mining_stats_epoch['kept_samples'].append(mining_stats.get('kept_samples', 0))
            else:
                # Loss padrão sem mining
                total_loss, loss_dict = loss_fn(
                    cls_pred, box_pred, landmark_pred,
                    cls_target, box_target, landmark_target, sample_type
                )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Registrar losses
            epoch_losses['total'].append(loss_dict['total'])
            epoch_losses['cls'].append(loss_dict['cls'])
            epoch_losses['box'].append(loss_dict['box'])
            epoch_losses['landmark'].append(loss_dict['landmark'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['cls']:.4f}",
                'box': f"{loss_dict['box']:.4f}",
                'lmk': f"{loss_dict['landmark']:.4f}"
            })
            
            # TensorBoard logging
            if global_step % Config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/total', loss_dict['total'], global_step)
                writer.add_scalar('Loss/cls', loss_dict['cls'], global_step)
                writer.add_scalar('Loss/box', loss_dict['box'], global_step)
                writer.add_scalar('Loss/landmark', loss_dict['landmark'], global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                if mining is not None and mining_stats:
                    writer.add_scalar('Mining/hard_ratio', mining_stats.get('hard_ratio', 0), global_step)
                    writer.add_scalar('Mining/kept_samples', mining_stats.get('kept_samples', 0), global_step)
            
            global_step += 1
        
        # ==================== EPOCH SUMMARY ====================
        avg_losses = {k: np.mean(v) if len(v) > 0 else 0 for k, v in epoch_losses.items()}
        
        print(f"\n[Epoch {epoch+1}] Summary:")
        print(f"  Total Loss: {avg_losses['total']:.4f}")
        print(f"  Cls Loss:   {avg_losses['cls']:.4f}")
        print(f"  Box Loss:   {avg_losses['box']:.4f}")
        print(f"  Lmk Loss:   {avg_losses['landmark']:.4f}")
        
        if mining is not None and len(mining_stats_epoch['hard_ratio']) > 0:
            avg_hard_ratio = np.mean(mining_stats_epoch['hard_ratio'])
            avg_kept = np.mean(mining_stats_epoch['kept_samples'])
            print(f"  Hard Ratio: {avg_hard_ratio:.3f}")
            print(f"  Kept Samples: {avg_kept:.1f}")
        
        # Learning rate step
        scheduler.step()
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0 or avg_losses['total'] < best_loss:
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, 'pnet', f'pnet_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint salvo: {checkpoint_path}")
            
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                best_path = os.path.join(Config.CHECKPOINT_DIR, 'pnet', 'pnet_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model atualizado! Loss: {best_loss:.4f}")
        
        print("-"*70)
    
    # ==================== FINALIZATION ====================
    final_path = Config.PNET_CHECKPOINT
    torch.save(model.state_dict(), final_path)
    print(f"\n[Finalizado] Model final salvo: {final_path}")
    print(f"[Finalizado] Best loss: {best_loss:.4f}")
    
    writer.close()
    
    print("\n" + "="*70)
    print("✓ TREINAMENTO DO P-NET CONCLUÍDO!")
    print("="*70)


if __name__ == "__main__":
    train_pnet()