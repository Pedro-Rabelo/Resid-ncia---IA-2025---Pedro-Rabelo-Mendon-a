import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.mobilenetv3_multitask import mobilenetv3_large_multitask
from utils.metrics import MarginCosineProduct
from utils.dataset_landmarks import ImageFolderWithLandmarks, create_validation_split_with_landmarks
from utils.multitask_loss import MultiTaskLossAdvanced
from utils.general import (
    setup_seed,
    AverageMeter,
    LOGGER,
    save_on_master
)
import evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="VGGFace2 Multi-task Training with Landmarks")
    
    # Dataset and Paths
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Path to VGGFace2 aligned images (e.g., data/train/vggface2_aligned_112x112)'
    )
    parser.add_argument(
        '--landmarks-json',
        type=str,
        required=True,
        help='Path to landmarks JSON file (e.g., data/train/vggface2_landmarks.json)'
    )
    
    # Model Settings
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )
    
    # Training Hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate (default: 0.1)'
    )
    
    # Learning Rate Scheduler
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 20, 25],
        help='Epochs to reduce learning rate (default: [10, 20, 25])'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Learning rate decay factor (default: 0.1)'
    )
    
    # Optimizer
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay (default: 5e-4)'
    )
    
    # Multi-task Learning
    parser.add_argument(
        '--landmark-weight',
        type=float,
        default=0.5,
        help='Weight for landmark loss (default: 0.5)'
    )
    parser.add_argument(
        '--use-wing-loss',
        action='store_true',
        help='Use Wing Loss for landmarks instead of SmoothL1'
    )
    
    # Dataset Split
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8 = 80%% train, 20%% val)'
    )
    parser.add_argument(
        '--min-images-per-class',
        type=int,
        default=2,
        help='Minimum images per identity to keep (default: 2)'
    )
    
    # Paths
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights/vggface2',
        help='Path to save model checkpoints (default: weights/vggface2)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    # Training
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loader workers (default: 8)'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Print frequency in batches (default: 100)'
    )
    
    # LFW Evaluation
    parser.add_argument(
        '--lfw-root',
        type=str,
        default='data/val',
        help='Path to LFW dataset for validation (default: data/val)'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=1,
        help='LFW evaluation frequency in epochs (default: 1 = every epoch)'
    )
    
    return parser.parse_args()


def train_one_epoch_multitask(
    model,
    classification_head,
    criterion_multitask,
    optimizer,
    data_loader,
    device,
    epoch,
    params
):
    """Training loop for one epoch with multi-task learning"""
    model.train()
    classification_head.train()
    
    losses_total = AverageMeter("Total Loss", ":6.3f")
    losses_cls = AverageMeter("Cls Loss", ":6.3f")
    losses_landmark = AverageMeter("Landmark Loss", ":6.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    batch_time = AverageMeter("Time", ":4.3f")
    
    start_time = time.time()
    last_batch_idx = len(data_loader) - 1
    
    for batch_idx, (images, targets, landmarks_gt) in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx
        
        # Move to device
        images = images.to(device)
        targets = targets.to(device)
        landmarks_gt = landmarks_gt.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with multi-task
        embeddings, landmarks_pred = model(images, return_landmarks=True)
        
        # Classification via MCP
        cls_output = classification_head(embeddings, targets)
        
        # Multi-task loss
        total_loss, loss_dict = criterion_multitask(
            cls_output, 
            landmarks_pred,
            targets,
            landmarks_gt
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(cls_output.data, 1)
        accuracy = (predicted == targets).float().mean() * 100
        
        # Update metrics
        losses_total.update(loss_dict['total'], images.size(0))
        losses_cls.update(loss_dict['classification'], images.size(0))
        losses_landmark.update(loss_dict['landmark'], images.size(0))
        accuracy_meter.update(accuracy.item(), images.size(0))
        batch_time.update(time.time() - start_time)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Log progress
        if batch_idx % params.print_freq == 0 or last_batch:
            lr = optimizer.param_groups[0]['lr']
            log = (
                f'Epoch: [{epoch}/{params.epochs}][{batch_idx:05d}/{len(data_loader):05d}] '
                f'Loss: {losses_total.avg:6.3f} (Cls: {losses_cls.avg:6.3f}, '
                f'Lmk: {losses_landmark.avg:6.3f}) '
                f'Acc: {accuracy_meter.avg:4.2f}% '
                f'LR: {lr:.5f} '
                f'Time: {batch_time.avg:4.3f}s'
            )
            LOGGER.info(log)
    
    # End-of-epoch summary
    log = (
        f'Epoch [{epoch}/{params.epochs}] Summary: '
        f'Total Loss: {losses_total.avg:6.3f}, '
        f'Cls Loss: {losses_cls.avg:6.3f}, '
        f'Landmark Loss: {losses_landmark.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%'
    )
    LOGGER.info(log)


def validate_lfw(model, device, lfw_root='data/val'):
    """Validates model on LFW dataset"""
    try:
        model.eval()
        accuracy, _ = evaluate.eval(model, device=device, lfw_root=lfw_root)
        model.train()
        return accuracy
    except Exception as e:
        LOGGER.warning(f"LFW validation failed: {e}")
        return 0.0


def main(params):
    # Setup
    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LOGGER.info("="*70)
    LOGGER.info("VGGFACE2 TRAINING - MULTI-TASK LEARNING")
    LOGGER.info("="*70)
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Root: {params.root}")
    LOGGER.info(f"Landmarks: {params.landmarks_json}")
    LOGGER.info(f"Batch size: {params.batch_size}")
    LOGGER.info(f"Epochs: {params.epochs}")
    LOGGER.info(f"Train/Val split: {params.train_split*100:.0f}%/{(1-params.train_split)*100:.0f}%")
    LOGGER.info("="*70 + "\n")
    
    # VGGFace2 has 8,631 identities
    EXPECTED_NUM_CLASSES = 8631
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Load dataset with landmarks
    LOGGER.info("Loading VGGFace2 dataset with landmarks...")
    full_dataset = ImageFolderWithLandmarks(
        root=params.root,
        landmarks_json=params.landmarks_json,
        transform=train_transform,
        min_images_per_class=params.min_images_per_class,
        database='VggFace2'
    )
    
    # Get actual number of classes
    num_classes = full_dataset.get_num_classes()
    LOGGER.info(f"Dataset loaded: {num_classes:,} classes")
    
    # Validation check
    if abs(num_classes - EXPECTED_NUM_CLASSES) > 100:
        LOGGER.warning(f"⚠️  Expected ~{EXPECTED_NUM_CLASSES:,} classes, got {num_classes:,}")
    
    # Split into train and validation (80/20)
    val_split = 1.0 - params.train_split
    train_dataset, val_dataset = create_validation_split_with_landmarks(
        full_dataset, 
        val_split=val_split
    )
    
    LOGGER.info(f"Training samples: {len(train_dataset):,}")
    LOGGER.info(f"Validation samples: {len(val_dataset):,}")
    LOGGER.info(f"Split ratio: {params.train_split*100:.1f}% / {val_split*100:.1f}%\n")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Model with multi-task learning
    LOGGER.info("Creating MobileNetV3-Large with multi-task learning...")
    model = mobilenetv3_large_multitask(embedding_dim=params.embedding_dim).to(device)
    
    # Classification head (MCP - CosFace)
    classification_head = MarginCosineProduct(
        in_features=params.embedding_dim,
        out_features=num_classes,
        s=30.0,
        m=0.40
    ).to(device)
    
    LOGGER.info(f"Model created: {params.embedding_dim}D embeddings, {num_classes:,} classes")
    
    # Multi-task loss
    criterion_multitask = MultiTaskLossAdvanced(
        landmark_weight=params.landmark_weight,
        use_wing_loss=params.use_wing_loss
    )
    
    loss_type = "Wing Loss" if params.use_wing_loss else "SmoothL1 Loss"
    LOGGER.info(f"Loss: Classification + Landmark ({loss_type}, weight={params.landmark_weight})\n")
    
    # Optimizer
    optimizer = torch.optim.SGD(
        [
            {'params': model.parameters()},
            {'params': classification_head.parameters()}
        ],
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=params.milestones,
        gamma=params.gamma
    )
    
    LOGGER.info(f"Optimizer: SGD (lr={params.lr}, momentum={params.momentum}, wd={params.weight_decay})")
    LOGGER.info(f"LR Schedule: MultiStepLR (milestones={params.milestones}, gamma={params.gamma})\n")
    
    # Resume from checkpoint
    start_epoch = 0
    best_lfw_accuracy = 0.0
    
    if params.checkpoint and os.path.isfile(params.checkpoint):
        LOGGER.info(f"Resuming from checkpoint: {params.checkpoint}")
        ckpt = torch.load(params.checkpoint, map_location="cpu")
        
        model.load_state_dict(ckpt['model'])
        classification_head.load_state_dict(ckpt['classification_head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        
        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        start_epoch = ckpt['epoch']
        best_lfw_accuracy = ckpt.get('best_lfw_accuracy', 0.0)
        
        LOGGER.info(f"Resumed from epoch {start_epoch}, best LFW: {best_lfw_accuracy:.4f}\n")
    
    # Create save directory
    os.makedirs(params.save_path, exist_ok=True)
    
    # Training loop
    LOGGER.info("="*70)
    LOGGER.info("STARTING TRAINING")
    LOGGER.info("="*70 + "\n")
    
    for epoch in range(start_epoch, params.epochs):
        # Train one epoch
        train_one_epoch_multitask(
            model,
            classification_head,
            criterion_multitask,
            optimizer,
            train_loader,
            device,
            epoch,
            params
        )
        
        # Step scheduler
        lr_scheduler.step()
        
        # Save last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'classification_head': classification_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_lfw_accuracy': best_lfw_accuracy,
            'num_classes': num_classes,
            'args': params
        }
        
        last_save_path = os.path.join(
            params.save_path,
            'mobilenetv3_vggface2_multitask_last.ckpt'
        )
        save_on_master(checkpoint, last_save_path)
        
        # LFW Evaluation
        if (epoch + 1) % params.eval_freq == 0:
            LOGGER.info(f"\nEvaluating on LFW (epoch {epoch+1})...")
            lfw_accuracy = validate_lfw(model, device, params.lfw_root)
            LOGGER.info(f"LFW Accuracy: {lfw_accuracy:.4f}\n")
            
            # Save best model
            if lfw_accuracy > best_lfw_accuracy:
                best_lfw_accuracy = lfw_accuracy
                checkpoint['best_lfw_accuracy'] = best_lfw_accuracy
                
                best_save_path = os.path.join(
                    params.save_path,
                    'mobilenetv3_vggface2_multitask_best.ckpt'
                )
                save_on_master(checkpoint, best_save_path)
                
                LOGGER.info(f"✅ New best LFW accuracy: {best_lfw_accuracy:.4f}")
                LOGGER.info(f"✅ Best model saved to: {best_save_path}\n")
    
    # Training completed
    LOGGER.info("="*70)
    LOGGER.info("TRAINING COMPLETED")
    LOGGER.info("="*70)
    LOGGER.info(f"Best LFW accuracy: {best_lfw_accuracy:.4f}")
    LOGGER.info(f"Models saved in: {params.save_path}")
    LOGGER.info("="*70)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)