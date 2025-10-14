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
from models.rnet import RNet
from utils.data_utils import parse_wider_face_annotation, parse_celeba_landmarks
from utils.bbox_utils import (compute_iou, nms, calibrate_box, convert_to_square, 
                              generate_bboxes, pad_bbox)
from config import Config


def detect_with_pnet_rnet(pnet, rnet, image, min_face_size=20):
    """
    Detecta faces usando P-Net + R-Net em cascata
    
    Args:
        pnet: modelo P-Net
        rnet: modelo R-Net
        image: numpy array [H, W, 3] RGB
        min_face_size: tamanho mínimo
    
    Returns:
        bboxes: detecções refinadas [N, 4]
    """
    h, w = image.shape[:2]
    
    # Stage 1: P-Net
    min_length = min(h, w)
    factor = 0.709
    scales = []
    m = 12 / min_face_size
    min_length *= m
    
    while min_length > 12:
        scales.append(m)
        min_length *= factor
        m *= factor
    
    all_boxes = []
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        img_resized = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_LINEAR)
        
        img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            cls_prob, box_reg, _ = pnet.predict(img_tensor)
        
        cls_prob = cls_prob.cpu().numpy().squeeze()
        box_reg = box_reg.cpu().numpy().squeeze()
        
        boxes = generate_bboxes(cls_prob, box_reg, scale, Config.PNET_THRESHOLD)
        if len(boxes) > 0:
            all_boxes.append(boxes)
    
    if len(all_boxes) == 0:
        return np.array([])
    
    all_boxes = np.vstack(all_boxes)
    keep = nms(all_boxes[:, :5], all_boxes[:, 4], 0.7)
    all_boxes = all_boxes[keep]
    
    bboxes = all_boxes[:, :4]
    offsets = all_boxes[:, 5:9]
    bboxes = calibrate_box(bboxes, offsets)
    bboxes = convert_to_square(bboxes)
    
    # Stage 2: R-Net
    if len(bboxes) == 0:
        return np.array([])
    
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(int)
        bbox_padded, pad = pad_bbox(bbox, image.shape)
        x1, y1, x2, y2 = bbox_padded
        
        crop = image[y1:y2+1, x1:x2+1]
        if pad != [0, 0, 0, 0]:
            crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_CONSTANT)
        
        crop_resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_LINEAR)
        img_tensor = (crop_resized.astype(np.float32) - 127.5) / 128.0
        patches.append(img_tensor)
    
    patches = np.stack(patches)
    patches = torch.from_numpy(patches).permute(0, 3, 1, 2).to(Config.DEVICE)
    
    with torch.no_grad():
        cls_prob, box_reg, _ = rnet.predict(patches)
    
    cls_prob = cls_prob.cpu().numpy()
    box_reg = box_reg.cpu().numpy()
    
    keep_idx = cls_prob > Config.RNET_THRESHOLD
    bboxes = bboxes[keep_idx]
    box_reg = box_reg[keep_idx]
    cls_prob = cls_prob[keep_idx]
    
    if len(bboxes) == 0:
        return np.array([])
    
    bboxes = calibrate_box(bboxes, box_reg)
    bboxes_with_scores = np.column_stack([bboxes, cls_prob])
    keep = nms(bboxes_with_scores[:, :5], bboxes_with_scores[:, 4], 0.7)
    bboxes = bboxes[keep]
    bboxes = convert_to_square(bboxes)
    
    return bboxes


def generate_onet_data():
    """
    Gera dados para O-Net usando P-Net + R-Net
    """
    print("\n" + "="*70)
    print("GERAÇÃO DE DADOS PARA O-NET")
    print("="*70)
    
    # Setup
    output_dir = os.path.join(Config.PROCESSED_DATA_DIR, 'onet')
    os.makedirs(output_dir, exist_ok=True)
    
    pos_dir = os.path.join(output_dir, 'positive')
    neg_dir = os.path.join(output_dir, 'negative')
    part_dir = os.path.join(output_dir, 'part')
    landmark_dir = os.path.join(output_dir, 'landmark')
    
    for d in [pos_dir, neg_dir, part_dir, landmark_dir]:
        os.makedirs(d, exist_ok=True)
    
    annotation_file = os.path.join(output_dir, 'onet_train.txt')
    
    # Carregar modelos
    print("\n[1/5] Carregando P-Net e R-Net...")
    
    if not os.path.exists(Config.PNET_CHECKPOINT):
        print(f"❌ P-Net não encontrado")
        return
    
    if not os.path.exists(Config.RNET_CHECKPOINT):
        print(f"❌ R-Net não encontrado")
        return
    
    pnet = PNet().to(Config.DEVICE)
    pnet.load_state_dict(torch.load(Config.PNET_CHECKPOINT, map_location=Config.DEVICE))
    pnet.eval()
    
    rnet = RNet().to(Config.DEVICE)
    rnet.load_state_dict(torch.load(Config.RNET_CHECKPOINT, map_location=Config.DEVICE))
    rnet.eval()
    
    print("  ✓ Modelos carregados")
    
    # Anotações
    print("\n[2/5] Carregando anotações...")
    
    wider_anno_file = os.path.join(
        Config.WIDER_FACE_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt'
    )
    wider_annos = parse_wider_face_annotation(wider_anno_file)
    print(f"  ✓ WIDER FACE: {len(wider_annos)}")
    
    celeba_landmark_file = os.path.join(Config.CELEBA_DIR, 'list_landmarks_celeba.txt')
    celeba_landmarks = {}
    if os.path.exists(celeba_landmark_file):
        celeba_landmarks = parse_celeba_landmarks(celeba_landmark_file)
        print(f"  ✓ CelebA: {len(celeba_landmarks)}")
    
    # Gerar samples
    print("\n[3/5] Gerando samples com P-Net + R-Net...")
    
    counters = {'positive': 0, 'negative': 0, 'part': 0, 'landmark': 0}
    annotations = []
    
    for anno in tqdm(wider_annos[:5000], desc="WIDER FACE"):
        img_path = os.path.join(Config.WIDER_FACE_DIR, 'WIDER_train', 'images', anno['image_path'])
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt_boxes = anno['bboxes']
        
        # Detectar com P-Net + R-Net
        detected_boxes = detect_with_pnet_rnet(pnet, rnet, img_rgb)
        
        if len(detected_boxes) == 0:
            continue
        
        for det_box in detected_boxes:
            x1, y1, x2, y2 = det_box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            ious = compute_iou(det_box, gt_boxes)
            max_iou = ious.max() if len(ious) > 0 else 0
            best_gt_idx = ious.argmax() if len(ious) > 0 else -1
            
            crop = img_rgb[y1:y2+1, x1:x2+1]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
            
            # POSITIVE
            if max_iou >= Config.IOU_POSITIVE and counters['positive'] < 40000:
                gt_box = gt_boxes[best_gt_idx]
                crop_w, crop_h = x2 - x1 + 1, y2 - y1 + 1
                
                offset_x1 = (gt_box[0] - x1) / crop_w
                offset_y1 = (gt_box[1] - y1) / crop_h
                offset_x2 = (gt_box[2] - x2) / crop_w
                offset_y2 = (gt_box[3] - y2) / crop_h
                
                save_path = os.path.join(pos_dir, f"{counters['positive']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                # ✅ CORREÇÃO: Converter para caminho relativo
                rel_path = os.path.relpath(save_path, output_dir)
                
                annotations.append(
                    f"{rel_path} 1 {offset_x1:.4f} {offset_y1:.4f} {offset_x2:.4f} {offset_y2:.4f} "
                    f"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
                )
                counters['positive'] += 1
            
            # PART
            elif max_iou >= Config.IOU_PART and counters['part'] < 40000:
                gt_box = gt_boxes[best_gt_idx]
                crop_w, crop_h = x2 - x1 + 1, y2 - y1 + 1
                
                offset_x1 = (gt_box[0] - x1) / crop_w
                offset_y1 = (gt_box[1] - y1) / crop_h
                offset_x2 = (gt_box[2] - x2) / crop_w
                offset_y2 = (gt_box[3] - y2) / crop_h
                
                save_path = os.path.join(part_dir, f"{counters['part']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                # ✅ CORREÇÃO: Converter para caminho relativo
                rel_path = os.path.relpath(save_path, output_dir)
                
                annotations.append(
                    f"{rel_path} 2 {offset_x1:.4f} {offset_y1:.4f} {offset_x2:.4f} {offset_y2:.4f} "
                    f"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
                )
                counters['part'] += 1
            
            # NEGATIVE
            elif max_iou < Config.IOU_NEGATIVE and counters['negative'] < 40000:
                save_path = os.path.join(neg_dir, f"{counters['negative']}.jpg")
                Image.fromarray(crop_resized).save(save_path)
                
                # ✅ CORREÇÃO: Converter para caminho relativo
                rel_path = os.path.relpath(save_path, output_dir)
                
                annotations.append(
                    f"{rel_path} 0 0 0 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
                )
                counters['negative'] += 1
    
    # Landmarks
    print("\n[4/5] Gerando landmarks do CelebA...")
    
    if len(celeba_landmarks) > 0:
        celeba_img_dir = os.path.join(Config.CELEBA_DIR, 'img_celeba')
        
        for img_name, landmarks in tqdm(list(celeba_landmarks.items())[:40000], desc="CelebA"):
            if counters['landmark'] >= 40000:
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
            crop_resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
            
            save_path = os.path.join(landmark_dir, f"{counters['landmark']}.jpg")
            Image.fromarray(crop_resized).save(save_path)
            
            # ✅ CORREÇÃO: Converter para caminho relativo
            rel_path = os.path.relpath(save_path, output_dir)
            
            lmk_str = ' '.join([f"{x:.4f}" for x in landmarks_norm])
            annotations.append(f"{rel_path} 3 0 0 0 0 {lmk_str}\n")
            
            counters['landmark'] += 1
    
    # Salvar
    print("\n[5/5] Salvando...")
    random.shuffle(annotations)
    
    with open(annotation_file, 'w') as f:
        f.writelines(annotations)
    
    print(f"  ✓ {annotation_file}")
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO O-NET")
    print("="*70)
    print(f"Total: {sum(counters.values()):,}")
    for k, v in counters.items():
        print(f"  • {k}: {v:,}")
    print("="*70)
    print("\n✓ GERAÇÃO O-NET CONCLUÍDA!")


if __name__ == "__main__":
    generate_onet_data()