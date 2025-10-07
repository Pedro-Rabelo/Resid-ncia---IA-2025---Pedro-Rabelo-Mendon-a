"""
MTCNN - Multi-task Cascaded Convolutional Networks
Script Principal para Treinamento e Inferência

Uso:
    python main.py --mode train --stage pnet
    python main.py --mode train --stage rnet  
    python main.py --mode train --stage onet
    python main.py --mode detect --image path/to/image.jpg
"""

import argparse
import os
import torch
from config import Config

def prepare_data(stage):
    """Prepara dados para um estágio específico"""
    print(f"\n{'='*60}")
    print(f"PREPARANDO DADOS PARA {stage.upper()}")
    print('='*60)
    
    if stage == 'pnet':
        from data_preprocessing.generate_pnet_data import generate_pnet_data
        generate_pnet_data()
    
    elif stage == 'rnet':
        from data_preprocessing.generate_rnet_data import generate_rnet_data
        generate_rnet_data()
    
    elif stage == 'onet':
        from data_preprocessing.generate_onet_data import generate_onet_data
        generate_onet_data()
    
    print(f"\n✓ Dados para {stage.upper()} preparados!")

def train_network(stage):
    """Treina uma rede específica"""
    print(f"\n{'='*60}")
    print(f"TREINANDO {stage.upper()}")
    print('='*60)
    
    if stage == 'pnet':
        from training.train_pnet import train_pnet
        train_pnet()
    
    elif stage == 'rnet':
        from training.train_rnet import train_rnet
        train_rnet()
    
    elif stage == 'onet':
        from training.train_onet import train_onet
        train_onet()
    
    print(f"\n✓ {stage.upper()} treinada com sucesso!")

def detect_faces(image_path, output_path=None):
    """Detecta faces em uma imagem"""
    print(f"\n{'='*60}")
    print("DETECÇÃO DE FACES")
    print('='*60)
    
    import cv2
    from inference.detector import MTCNNDetector
    from inference.visualization import draw_detections
    
    # Carregar imagem
    print(f"\nCarregando imagem: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inicializar detector
    detector = MTCNNDetector(
        pnet_path=Config.PNET_CHECKPOINT,
        rnet_path=Config.RNET_CHECKPOINT,
        onet_path=Config.ONET_CHECKPOINT
    )
    
    # Detectar
    bboxes, scores, landmarks = detector.detect_faces(image_rgb)
    
    print(f"\n✓ {len(bboxes)} faces detectadas!")
    
    # Visualizar
    if output_path:
        result_img = draw_detections(image_rgb, bboxes, scores, landmarks)
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_img_bgr)
        print(f"✓ Resultado salvo em: {output_path}")
    
    return bboxes, scores, landmarks

def full_pipeline():
    """Pipeline completo: preparação de dados + treinamento"""
    print("\n" + "="*60)
    print("PIPELINE COMPLETO MTCNN")
    print("="*60)
    print("\nEste processo pode levar várias horas/dias dependendo do hardware.")
    print("Você será guiado através de cada estágio.\n")
    
    response = input("Deseja continuar? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelado.")
        return
    
    # Stage 1: P-Net
    print("\n" + "▶"*30)
    print("STAGE 1: P-NET")
    print("▶"*30)
    
    prepare_data('pnet')
    train_network('pnet')
    
    # Stage 2: R-Net
    print("\n" + "▶"*30)
    print("STAGE 2: R-NET")
    print("▶"*30)
    
    prepare_data('rnet')
    train_network('rnet')
    
    # Stage 3: O-Net
    print("\n" + "▶"*30)
    print("STAGE 3: O-NET")
    print("▶"*30)
    
    prepare_data('onet')
    train_network('onet')
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETO FINALIZADO!")
    print("="*60)
    print("\nModelos salvos em:")
    print(f"  - P-Net: {Config.PNET_CHECKPOINT}")
    print(f"  - R-Net: {Config.RNET_CHECKPOINT}")
    print(f"  - O-Net: {Config.ONET_CHECKPOINT}")

def main():
    parser = argparse.ArgumentParser(description='MTCNN Implementation')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['prepare', 'train', 'detect', 'full'],
                       help='Modo de operação')
    
    parser.add_argument('--stage', type=str, default=None,
                       choices=['pnet', 'rnet', 'onet'],
                       help='Estágio específico (para prepare/train)')
    
    parser.add_argument('--image', type=str, default=None,
                       help='Caminho da imagem para detecção')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Caminho de saída para imagem com detecções')
    
    args = parser.parse_args()
    
    # Criar estrutura de diretórios
    Config.create_dirs()
    Config.print_config()
    
    # Executar modo selecionado
    if args.mode == 'prepare':
        if args.stage is None:
            print("Erro: --stage é obrigatório para mode=prepare")
            return
        prepare_data(args.stage)
    
    elif args.mode == 'train':
        if args.stage is None:
            print("Erro: --stage é obrigatório para mode=train")
            return
        train_network(args.stage)
    
    elif args.mode == 'detect':
        if args.image is None:
            print("Erro: --image é obrigatório para mode=detect")
            return
        
        output = args.output if args.output else args.image.replace('.', '_result.')
        detect_faces(args.image, output)
    
    elif args.mode == 'full':
        full_pipeline()

if __name__ == "__main__":
    main()