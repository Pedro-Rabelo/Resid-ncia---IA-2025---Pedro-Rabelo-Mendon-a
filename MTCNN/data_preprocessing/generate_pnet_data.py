"""
========================================================================================
ARQUIVO 1: data_preprocessing/generate_pnet_data.py
========================================================================================
"""

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import parse_wider_face_annotation, parse_celeba_landmarks
from utils.bbox_utils import compute_iou
from config import Config

def generate_pnet_data():
    """
    Gera dados de treinamento para P-Net
    
    Processo:
    1. Carregar anotações do WIDER FACE e CelebA
    2. Para cada imagem do WIDER FACE:
       - Gerar negatives (IoU < 0.3)
       - Gerar positives (IoU > 0.65)
       - Gerar part faces (0.4 < IoU < 0.65)
    3. Para imagens do CelebA com landmarks:
       - Gerar landmark faces
    4. Salvar crops e anotações
    """
    print("\n" + "="*70)
    print("GERAÇÃO DE DADOS PARA P-NET")
    print("="*70)
    
    # ==================== PATHS ====================
    output_dir = os.path.join(Config.PROCESSED_DATA_DIR, 'pnet')
    os.makedirs(output_dir, exist_ok=True)
    
    # Subdiretórios para cada tipo
    pos_dir = os.path.join(output_dir, 'positive')
    neg_dir = os.path.join(output_dir, 'negative')
    part_dir = os.path.join(output_dir, 'part')
    landmark_dir = os.path.join(output_dir, 'landmark')
    
    for d in [pos_dir, neg_dir, part_dir, landmark_dir]:
        os.makedirs(d, exist_ok=True)
    
    annotation_file = os.path.join(output_dir, 'pnet_train.txt')
    
    # ==================== LOAD ANNOTATIONS ====================
    print("\n[1/4] Carregando anotações...")
    
    wider_anno_file = os.path.join(
        Config.WIDER_FACE_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt'
    )
    
    if not os.path.exists(wider_anno_file):
        print(f"❌ Erro: Arquivo não encontrado: {wider_anno_file}")
        print("Por favor, baixe o WIDER FACE dataset.")
        return
    
    wider_annos = parse_wider_face_annotation(wider_anno_file)
    print(f"  ✓ WIDER FACE: {len(wider_annos)} imagens")
    
    celeba_landmark_file = os.path.join(
        Config.CELEBA_DIR, 'list_landmarks_celeba.txt'
    )
    
    celeba_landmarks = {}
    if os.path.exists(celeba_landmark_file):
        celeba_landmarks = parse_celeba_landmarks(celeba_landmark_file)
        print(f"  ✓ CelebA: {len(celeba_landmarks)} imagens com landmarks")
    else:
        print(f"  ⚠ CelebA landmarks não encontrado, será pulado")
    
    # ==================== GENERATE SAMPLES ====================
    print("\n[2/4] Gerando samples do WIDER FACE...")
    
    counters = {'positive': 0, 'negative': 0, 'part': 0, 'landmark': 0}
    annotations = []
    
    for anno_idx, anno in enumerate(tqdm(wider_annos, desc="Processing WIDER FACE")):
        img_path = os.path.join(Config.WIDER_FACE_DIR, 'WIDER_train', anno['image_path'])
        
        if not os.path.exists(img_path):
            continue
        
        # Carregar imagem
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        gt_boxes = anno['bboxes']
        num_boxes = len(gt_boxes)
        
        # ========== NEGATIVES ==========
        neg_count = 0
        attempts = 0
        max_attempts = 100
        
        while neg_count < Config.PNET_SAMPLES_PER_IMAGE['negative'] and attempts < max_attempts:
            attempts += 1
            
            # Random crop
            size = random.randint(12, min(w, h) // 2)
            x1 = random.randint(0, max(0, w - size))
            y1 = random.randint(0, max(0, h - size))
            x2 = x1 + size
            y2 = y1 + size
            
            crop_box = np.array([x1, y1, x2, y2])
            
            # Calcular IoU com todas as GT boxes
            ious = compute_iou(crop_box, gt_boxes)
            max_iou = ious.max() if len(ious) > 0 else 0
            
            # Aceitar se IoU < 0.3 (negative)
            if max_iou < Config.IOU_NEGATIVE:
                # Crop e resize
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                crop_resized = cv2.resize(crop, (12, 12), interpolation=cv2.INTER_LINEAR)
                
                # Salvar
                save_path = os.path.join(neg_dir, f"{counters['negative']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                # Anotação: img_path class x1 y1 x2 y2
                annotations.append(f"{save_path} 0 0 0 0 0\n")
                
                counters['negative'] += 1
                neg_count += 1
        
        # ========== POSITIVES e PART FACES ==========
        for gt_box in gt_boxes:
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            gt_w = gt_x2 - gt_x1 + 1
            gt_h = gt_y2 - gt_y1 + 1
            
            # Ignorar faces muito pequenas
            if gt_w < 20 or gt_h < 20:
                continue
            
            # Gerar crops ao redor da GT box
            for _ in range(max(1, Config.PNET_SAMPLES_PER_IMAGE['positive'] // num_boxes)):
                # Random offset
                offset_x = random.uniform(-0.2, 0.2) * gt_w
                offset_y = random.uniform(-0.2, 0.2) * gt_h
                offset_w = random.uniform(-0.2, 0.2) * gt_w
                offset_h = random.uniform(-0.2, 0.2) * gt_h
                
                x1 = int(max(0, gt_x1 + offset_x))
                y1 = int(max(0, gt_y1 + offset_y))
                x2 = int(min(w - 1, gt_x2 + offset_w))
                y2 = int(min(h - 1, gt_y2 + offset_h))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop_box = np.array([x1, y1, x2, y2])
                
                # Calcular IoU
                iou = compute_iou(crop_box, gt_boxes).max()
                
                # Calcular offsets normalizados
                crop_w = x2 - x1 + 1
                crop_h = y2 - y1 + 1
                
                offset_x1 = (gt_x1 - x1) / crop_w
                offset_y1 = (gt_y1 - y1) / crop_h
                offset_x2 = (gt_x2 - x2) / crop_w
                offset_y2 = (gt_y2 - y2) / crop_h
                
                # Crop
                crop = img_rgb[y1:y2+1, x1:x2+1]
                if crop.size == 0:
                    continue
                
                crop_resized = cv2.resize(crop, (12, 12), interpolation=cv2.INTER_LINEAR)
                
                # Classificar como positive ou part
                if iou >= Config.IOU_POSITIVE:
                    # POSITIVE
                    if counters['positive'] >= 100000:  # Limitar
                        continue
                    
                    save_path = os.path.join(pos_dir, f"{counters['positive']}.jpg")
                    Image.fromarray(crop_resized).save(save_path)
                    
                    # Anotação: img_path class offset_x1 offset_y1 offset_x2 offset_y2
                    annotations.append(
                        f"{save_path} 1 {offset_x1:.4f} {offset_y1:.4f} "
                        f"{offset_x2:.4f} {offset_y2:.4f}\n"
                    )
                    
                    counters['positive'] += 1
                
                elif iou >= Config.IOU_PART_MIN:
                    # PART FACE
                    if counters['part'] >= 100000:
                        continue
                    
                    save_path = os.path.join(part_dir, f"{counters['part']}.jpg")
                    Image.fromarray(crop_resized).save(save_path)
                    
                    annotations.append(
                        f"{save_path} 2 {offset_x1:.4f} {offset_y1:.4f} "
                        f"{offset_x2:.4f} {offset_y2:.4f}\n"
                    )
                    
                    counters['part'] += 1
    
    # ==================== GENERATE LANDMARK SAMPLES ====================
    print("\n[3/4] Gerando samples de landmarks do CelebA...")
    
    if len(celeba_landmarks) > 0:
        celeba_img_dir = os.path.join(Config.CELEBA_DIR, 'img_celeba')
        
        for img_name, landmarks in tqdm(list(celeba_landmarks.items())[:50000], 
                                        desc="Processing CelebA"):
            if counters['landmark'] >= 50000:
                break
                
            img_path = os.path.join(celeba_img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Landmarks: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
            # Calcular bounding box a partir dos landmarks
            x_coords = landmarks[0::2]
            y_coords = landmarks[1::2]
            
            x1 = int(max(0, x_coords.min() - 10))
            y1 = int(max(0, y_coords.min() - 10))
            x2 = int(min(w - 1, x_coords.max() + 10))
            y2 = int(min(h - 1, y_coords.max() + 10))
            
            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            
            # Ignorar faces muito pequenas
            if box_w < 20 or box_h < 20:
                continue
            
            # Normalizar landmarks
            landmarks_norm = landmarks.copy()
            landmarks_norm[0::2] = (landmarks[0::2] - x1) / box_w
            landmarks_norm[1::2] = (landmarks[1::2] - y1) / box_h
            
            # Crop e resize
            crop = img_rgb[y1:y2+1, x1:x2+1]
            if crop.size == 0:
                continue
                
            crop_resized = cv2.resize(crop, (12, 12), interpolation=cv2.INTER_LINEAR)
            
            # Salvar
            save_path = os.path.join(landmark_dir, f"{counters['landmark']}.jpg")
            Image.fromarray(crop_resized).save(save_path)
            
            # Anotação: img_path class 0 0 0 0 lmk1_x lmk1_y ... lmk5_x lmk5_y
            lmk_str = ' '.join([f"{x:.4f}" for x in landmarks_norm])
            annotations.append(f"{save_path} 3 0 0 0 0 {lmk_str}\n")
            
            counters['landmark'] += 1
    
    # ==================== SAVE ANNOTATIONS ====================
    print("\n[4/4] Salvando anotações...")
    
    # Shuffle annotations
    random.shuffle(annotations)
    
    with open(annotation_file, 'w') as f:
        f.writelines(annotations)
    
    print(f"  ✓ Anotações salvas em: {annotation_file}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("RESUMO DA GERAÇÃO DE DADOS")
    print("="*70)
    print(f"Total de samples gerados: {sum(counters.values()):,}")
    print(f"  • Positives:  {counters['positive']:,}")
    print(f"  • Negatives:  {counters['negative']:,}")
    print(f"  • Part faces: {counters['part']:,}")
    print(f"  • Landmarks:  {counters['landmark']:,}")
    print("\nDistribuição:")
    total = sum(counters.values())
    for key, count in counters.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {key:10s}: {pct:5.1f}%")
    print("="*70)
    
    print("\n✓ GERAÇÃO DE DADOS PARA P-NET CONCLUÍDA!")


if __name__ == "__main__":
    generate_pnet_data()