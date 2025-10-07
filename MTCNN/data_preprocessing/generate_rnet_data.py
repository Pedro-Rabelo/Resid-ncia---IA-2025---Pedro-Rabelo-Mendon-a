import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pnet import PNet
from utils.data_utils import parse_wider_face_annotation, parse_celeba_landmarks
from utils.bbox_utils import compute_iou, nms, calibrate_box, convert_to_square, generate_bboxes
from config import Config

def detect_with_pnet(pnet, image, min_face_size=20):
    """
    Detecta faces usando P-Net treinado
    
    Args:
        pnet: modelo P-Net carregado
        image: numpy array [H, W, 3] RGB
        min_face_size: tamanho mínimo de face
    
    Returns:
        bboxes: numpy array [N, 4] - detecções do P-Net
    """
    h, w = image.shape[:2]
    
    # Calcular escalas
    min_length = min(h, w)
    factor = 0.709
    
    scales = []
    m = 12 / min_face_size
    min_length *= m
    
    while min_length > 12:
        scales.append(m)
        min_length *= factor
        m *= factor
    
    # Processar cada escala
    all_boxes = []
    
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        img_resized = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_LINEAR)
        
        # Normalizar
        img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(Config.DEVICE)
        
        # Forward
        with torch.no_grad():
            cls_prob, box_reg, _ = pnet.predict(img_tensor)
        
        cls_prob = cls_prob.cpu().numpy().squeeze()
        box_reg = box_reg.cpu().numpy().squeeze()
        
        # Gerar boxes
        boxes = generate_bboxes(cls_prob, box_reg, scale, Config.PNET_THRESHOLD)
        
        if len(boxes) > 0:
            all_boxes.append(boxes)
    
    if len(all_boxes) == 0:
        return np.array([])
    
    # Concatenar e aplicar NMS
    all_boxes = np.vstack(all_boxes)
    keep = nms(all_boxes[:, :5], all_boxes[:, 4], Config.PNET_NMS_THRESHOLD)
    all_boxes = all_boxes[keep]
    
    # Calibrar
    bboxes = all_boxes[:, :4]
    offsets = all_boxes[:, 5:9]
    bboxes = calibrate_box(bboxes, offsets)
    bboxes = convert_to_square(bboxes)
    
    return bboxes


def generate_rnet_data():
    """
    Gera dados de treinamento para R-Net usando P-Net
    """
    print("\n" + "="*70)
    print("GERAÇÃO DE DADOS PARA R-NET")
    print("="*70)
    
    # Setup
    output_dir = os.path.join(Config.PROCESSED_DATA_DIR, 'rnet')
    os.makedirs(output_dir, exist_ok=True)
    
    pos_dir = os.path.join(output_dir, 'positive')
    neg_dir = os.path.join(output_dir, 'negative')
    part_dir = os.path.join(output_dir, 'part')
    landmark_dir = os.path.join(output_dir, 'landmark')
    
    for d in [pos_dir, neg_dir, part_dir, landmark_dir]:
        os.makedirs(d, exist_ok=True)
    
    annotation_file = os.path.join(output_dir, 'rnet_train.txt')
    
    # Carregar P-Net
    print("\n[1/5] Carregando P-Net treinado...")
    
    if not os.path.exists(Config.PNET_CHECKPOINT):
        print(f"❌ Erro: P-Net não encontrado: {Config.PNET_CHECKPOINT}")
        return
    
    pnet = PNet().to(Config.DEVICE)
    pnet.load_state_dict(torch.load(Config.PNET_CHECKPOINT, map_location=Config.DEVICE))
    pnet.eval()
    print("  ✓ P-Net carregado")
    
    # Carregar anotações
    print("\n[2/5] Carregando anotações...")
    
    wider_anno_file = os.path.join(
        Config.WIDER_FACE_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt'
    )
    
    wider_annos = parse_wider_face_annotation(wider_anno_file)
    print(f"  ✓ WIDER FACE: {len(wider_annos)} imagens")
    
    celeba_landmark_file = os.path.join(Config.CELEBA_DIR, 'list_landmarks_celeba.txt')
    celeba_landmarks = {}
    if os.path.exists(celeba_landmark_file):
        celeba_landmarks = parse_celeba_landmarks(celeba_landmark_file)
        print(f"  ✓ CelebA: {len(celeba_landmarks)} imagens")
    
    # Gerar samples
    print("\n[3/5] Gerando samples usando P-Net...")
    
    counters = {'positive': 0, 'negative': 0, 'part': 0, 'landmark': 0}
    annotations = []
    
    for anno in tqdm(wider_annos[:5000], desc="WIDER FACE"):
        img_path = os.path.join(Config.WIDER_FACE_DIR, 'WIDER_train', anno['image_path'])
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt_boxes = anno['bboxes']
        
        # Detectar com P-Net
        pnet_boxes = detect_with_pnet(pnet, img_rgb)
        
        if len(pnet_boxes) == 0:
            continue
        
        # Processar detecções
        for pnet_box in pnet_boxes:
            x1, y1, x2, y2 = pnet_box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            ious = compute_iou(pnet_box, gt_boxes)
            max_iou = ious.max() if len(ious) > 0 else 0
            best_gt_idx = ious.argmax() if len(ious) > 0 else -1
            
            crop = img_rgb[y1:y2+1, x1:x2+1]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_LINEAR)
            
            if max_iou >= Config.IOU_POSITIVE and counters['positive'] < 50000:
                gt_box = gt_boxes[best_gt_idx]
                crop_w, crop_h = x2 - x1 + 1, y2 - y1 + 1
                
                offset_x1 = (gt_box[0] - x1) / crop_w
                offset_y1 = (gt_box[1] - y1) / crop_h
                offset_x2 = (gt_box[2] - x2) / crop_w
                offset_y2 = (gt_box[3] - y2) / crop_h
                
                save_path = os.path.join(pos_dir, f"{counters['positive']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                annotations.append(
                    f"{save_path} 1 {offset_x1:.4f} {offset_y1:.4f} "
                    f"{offset_x2:.4f} {offset_y2:.4f}\n"
                )
                counters['positive'] += 1
            
            elif max_iou >= Config.IOU_PART_MIN and counters['part'] < 50000:
                gt_box = gt_boxes[best_gt_idx]
                crop_w, crop_h = x2 - x1 + 1, y2 - y1 + 1
                
                offset_x1 = (gt_box[0] - x1) / crop_w
                offset_y1 = (gt_box[1] - y1) / crop_h
                offset_x2 = (gt_box[2] - x2) / crop_w
                offset_y2 = (gt_box[3] - y2) / crop_h
                
                save_path = os.path.join(part_dir, f"{counters['part']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                annotations.append(
                    f"{save_path} 2 {offset_x1:.4f} {offset_y1:.4f} "
                    f"{offset_x2:.4f} {offset_y2:.4f}\n"
                )
                counters['part'] += 1
            
            elif max_iou < Config.IOU_NEGATIVE and counters['negative'] < 50000:
                save_path = os.path.join(neg_dir, f"{counters['negative']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                annotations.append(f"{save_path} 0 0 0 0 0\n")
                counters['negative'] += 1
    
    # Landmarks
    print("\n[4/5] Gerando landmarks...")
    
    if len(celeba_landmarks) > 0:
        celeba_img_dir = os.path.join(Config.CELEBA_DIR, 'img_celeba')
        
        for img_name, landmarks in tqdm(list(celeba_landmarks.items())[:30000], desc="CelebA"):
            if counters['landmark'] >= 30000:
                break
            
            img_path = os.path.join(celeba_img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            x_coords = landmarks[0::2]
            y_coords = landmarks[1::2]
            
            x1 = int(max(0, x_coords.min() - 10))
            y1 = int(max(0, y_coords.min() - 10))
            x2 = int(min(w-1, x_coords.max() + 10))
            y2 = int(min(h-1, y_coords.max() + 10))
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            
            landmarks_norm = landmarks.copy()
            landmarks_norm[0::2] = (landmarks[0::2] - x1) / (x2 - x1 + 1)
            landmarks_norm[1::2] = (landmarks[1::2] - y1) / (y2 - y1 + 1)
            
            crop = img_rgb[y1:y2+1, x1:x2+1]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_LINEAR)
            
            save_path = os.path.join(landmark_dir, f"{counters['landmark']}.jpg")
            Image.fromarray(crop_resized).save(save_path)
            
            lmk_str = ' '.join([f"{x:.4f}" for x in landmarks_norm])
            annotations.append(f"{save_path} 3 0 0 0 0 {lmk_str}\n")
            
            counters['landmark'] += 1
    
    # Salvar
    print("\n[5/5] Salvando anotações...")
    random.shuffle(annotations)
    
    with open(annotation_file, 'w') as f:
        f.writelines(annotations)
    
    print(f"  ✓ Anotações: {annotation_file}")
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO R-NET")
    print("="*70)
    print(f"Total: {sum(counters.values()):,}")
    for k, v in counters.items():
        print(f"  • {k}: {v:,}")
    print("="*70)
    print("\n✓ GERAÇÃO R-NET CONCLUÍDA!")


if __name__ == "__main__":
    generate_rnet_data()