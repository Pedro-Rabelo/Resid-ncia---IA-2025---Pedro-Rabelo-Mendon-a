import os
import torch

class Config:
    """Configurações globais para o projeto MTCNN"""
    
    # ==================== PATHS ====================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    
    # Datasets
    WIDER_FACE_DIR = os.path.join(RAW_DATA_DIR, 'WIDER_FACE')
    CELEBA_DIR = os.path.join(RAW_DATA_DIR, 'CelebA')
    
    # ==================== DEVICE ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ==================== DATA GENERATION ====================
    # IoU thresholds para classificação de samples
    IOU_NEGATIVE = 0.3    # IoU < 0.3: negative
    IOU_POSITIVE = 0.65   # IoU > 0.65: positive
    IOU_PART_MIN = 0.4    # 0.4 < IoU < 0.65: part face
    IOU_PART_MAX = 0.65
    
    # Número de samples a gerar por tipo
    PNET_SAMPLES_PER_IMAGE = {
        'positive': 50,
        'negative': 50,
        'part': 50,
        'landmark': 10
    }
    
    RNET_SAMPLES_PER_IMAGE = {
        'positive': 20,
        'negative': 60,
        'part': 20,
        'landmark': 5
    }
    
    ONET_SAMPLES_PER_IMAGE = {
        'positive': 20,
        'negative': 60,
        'part': 20,
        'landmark': 10
    }
    
    # ==================== P-NET CONFIG ====================
    PNET_INPUT_SIZE = 12
    PNET_BATCH_SIZE = 512
    PNET_EPOCHS = 30
    PNET_LR = 0.001
    PNET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'pnet', 'pnet_final.pth')
    
    # Loss weights para P-Net
    PNET_CLS_WEIGHT = 1.0      # α1: classificação
    PNET_BOX_WEIGHT = 0.5      # α2: bounding box regression
    PNET_LANDMARK_WEIGHT = 0.5 # α3: landmark localization
    
    # ==================== R-NET CONFIG ====================
    RNET_INPUT_SIZE = 24
    RNET_BATCH_SIZE = 256
    RNET_EPOCHS = 25
    RNET_LR = 0.001
    RNET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'rnet', 'rnet_final.pth')
    
    # Loss weights para R-Net
    RNET_CLS_WEIGHT = 1.0
    RNET_BOX_WEIGHT = 0.5
    RNET_LANDMARK_WEIGHT = 0.5
    
    # ==================== O-NET CONFIG ====================
    ONET_INPUT_SIZE = 48
    ONET_BATCH_SIZE = 128
    ONET_EPOCHS = 25
    ONET_LR = 0.001
    ONET_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'onet', 'onet_final.pth')
    
    # Loss weights para O-Net (maior peso para landmarks)
    ONET_CLS_WEIGHT = 1.0
    ONET_BOX_WEIGHT = 0.5
    ONET_LANDMARK_WEIGHT = 1.0  # Peso maior para precisão de landmarks
    
    # ==================== HARD SAMPLE MINING ====================
    HARD_SAMPLE_RATIO = 0.7  # Top 70% das losses
    ONLINE_MINING = True      # Ativar hard sample mining
    
    # ==================== INFERENCE CONFIG ====================
    # P-Net inference
    PNET_THRESHOLD = 0.6
    PNET_NMS_THRESHOLD = 0.7
    PNET_STRIDE = 2
    PNET_CELL_SIZE = 12
    
    # R-Net inference
    RNET_THRESHOLD = 0.7
    RNET_NMS_THRESHOLD = 0.7
    
    # O-Net inference
    ONET_THRESHOLD = 0.7
    ONET_NMS_THRESHOLD = 0.7
    
    # Pyramid scales para multi-scale detection
    MIN_FACE_SIZE = 20
    SCALE_FACTOR = 0.709
    
    # ==================== AUGMENTATION ====================
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.5
    
    # Random crop ranges
    CROP_SCALE_RANGE = (0.8, 1.2)
    CROP_SHIFT_RANGE = (-0.1, 0.1)
    
    # Color jitter
    BRIGHTNESS_RANGE = 0.2
    CONTRAST_RANGE = 0.2
    SATURATION_RANGE = 0.2
    HUE_RANGE = 0.1
    
    # ==================== LOGGING ====================
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 5  # Salvar checkpoint a cada N epochs
    TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, 'runs')

    # ==================== CONFIGURAÇÃO RÁPIDA PARA TESTE ====================
    # Para teste rápido, reduzir epochs
    PNET_EPOCHS = 10  # Original: 30
    RNET_EPOCHS = 8   # Original: 25
    ONET_EPOCHS = 8   # Original: 25
    
    # Limitar samples (opcional, para teste muito rápido)
    # Adicionar no final da classe:
    QUICK_TEST = True  # Ativar modo teste rápido
    MAX_IMAGES = 2000  # Processar apenas 2000 imagens (vs 12000)
    
    @classmethod
    def create_dirs(cls):
        """Cria todos os diretórios necessários"""
        dirs = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            os.path.join(cls.PROCESSED_DATA_DIR, 'pnet'),
            os.path.join(cls.PROCESSED_DATA_DIR, 'rnet'),
            os.path.join(cls.PROCESSED_DATA_DIR, 'onet'),
            cls.CHECKPOINT_DIR,
            os.path.join(cls.CHECKPOINT_DIR, 'pnet'),
            os.path.join(cls.CHECKPOINT_DIR, 'rnet'),
            os.path.join(cls.CHECKPOINT_DIR, 'onet'),
            cls.TENSORBOARD_DIR
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print("✓ Estrutura de diretórios criada com sucesso!")
    
    @classmethod
    def print_config(cls):
        """Imprime configurações principais"""
        print("\n" + "="*60)
        print("CONFIGURAÇÕES DO MTCNN")
        print("="*60)
        print(f"Device: {cls.DEVICE}")
        print(f"P-Net Input: {cls.PNET_INPUT_SIZE}x{cls.PNET_INPUT_SIZE}")
        print(f"R-Net Input: {cls.RNET_INPUT_SIZE}x{cls.RNET_INPUT_SIZE}")
        print(f"O-Net Input: {cls.ONET_INPUT_SIZE}x{cls.ONET_INPUT_SIZE}")
        print(f"Hard Sample Mining: {'Ativado' if cls.ONLINE_MINING else 'Desativado'}")
        print(f"Hard Sample Ratio: {cls.HARD_SAMPLE_RATIO}")
        print("="*60 + "\n")


if __name__ == "__main__":
    Config.create_dirs()
    Config.print_config()