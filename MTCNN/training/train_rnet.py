import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnet import RNet
from models.losses import RNetLoss
from utils.data_utils import MTCNNDataset
from training.hard_sample_mining import OnlineHardSampleMining
from config import Config


def train_rnet():
    """Treina R-Net com Online Hard Sample Mining"""
    print("\n" + "="*70)
    print("TREINAMENTO DO R-NET")
    print("="*70)
    
    device = Config.DEVICE
    print(f"\n[Setup] Device: {device}")
    
    # Dataset
    train_anno_file = os.path.join(Config.PROCESSED_DATA_DIR, 'rnet', 'rnet_train.txt')
    
    if not os.path.exists(train_anno_file):
        print(f"\n❌ Erro: {train_anno_file} não encontrado")
        print("Execute: python main.py --mode prepare --stage rnet")
        return
    
    print(f"\n[Dataset] Carregando dados...")
    train_dataset = MTCNNDataset(
        annotation_file=train_anno_file,
        input_size=Config.RNET_INPUT_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.RNET_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    
    print(f"  Samples: {len(train_dataset):,}")
    print(f"  Batches: {len(train_loader):,}")
    
    # Model
    print(f"\n[Model] Inicializando R-Net...")
    model = RNet().to(device)
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss e Optimizer
    loss_fn = RNetLoss(
        cls_weight=Config.RNET_CLS_WEIGHT,
        box_weight=Config.RNET_BOX_WEIGHT,
        landmark_weight=Config.RNET_LANDMARK_WEIGHT
    )
    
    optimizer = optim.Adam(model.parameters(), lr=Config.RNET_LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 18], gamma=0.1)
    
    # Hard sample mining
    mining = OnlineHardSampleMining(ratio=Config.HARD_SAMPLE_RATIO) if Config.ONLINE_MINING else None
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(Config.TENSORBOARD_DIR, 'rnet'))
    
    # Training loop
    print(f"\n[Training] Iniciando...")
    print(f"  Epochs: {Config.RNET_EPOCHS}")
    print("-"*70)
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(Config.RNET_EPOCHS):
        model.train()
        epoch_losses = {'total': [], 'cls': [], 'box': [], 'landmark': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.RNET_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            cls_target = batch['cls_target'].to(device)
            box_target = batch['box_target'].to(device)
            landmark_target = batch['landmark_target'].to(device)
            sample_type = batch['sample_type'].to(device)
            
            cls_pred, box_pred, landmark_pred = model(images)
            
            if mining is not None:
                total_loss, loss_dict, _ = mining.compute_loss_with_mining(
                    loss_fn, cls_pred, box_pred, landmark_pred,
                    cls_target, box_target, landmark_target, sample_type
                )
            else:
                total_loss, loss_dict = loss_fn(
                    cls_pred, box_pred, landmark_pred,
                    cls_target, box_target, landmark_target, sample_type
                )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
            
            if global_step % Config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/total', loss_dict['total'], global_step)
                writer.add_scalar('Loss/cls', loss_dict['cls'], global_step)
                writer.add_scalar('Loss/box', loss_dict['box'], global_step)
                writer.add_scalar('Loss/landmark', loss_dict['landmark'], global_step)
            
            global_step += 1
        
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        print(f"\n[Epoch {epoch+1}] Loss: {avg_losses['total']:.4f}")
        
        scheduler.step()
        
        if (epoch + 1) % Config.SAVE_INTERVAL == 0 or avg_losses['total'] < best_loss:
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, 'rnet', f'rnet_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            
            if avg_losses['total'] < best_loss:
                best_loss = avg_losses['total']
                best_path = os.path.join(Config.CHECKPOINT_DIR, 'rnet', 'rnet_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model: {best_loss:.4f}")
        
        print("-"*70)
    
    final_path = Config.RNET_CHECKPOINT
    torch.save(model.state_dict(), final_path)
    print(f"\n[Finalizado] Model salvo: {final_path}")
    print(f"[Finalizado] Best loss: {best_loss:.4f}")
    
    writer.close()
    print("\n✓ TREINAMENTO DO R-NET CONCLUÍDO!")


if __name__ == "__main__":
    train_rnet()