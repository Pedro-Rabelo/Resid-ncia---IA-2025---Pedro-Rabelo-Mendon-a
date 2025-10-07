import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class MTCNNDataset(Dataset):
    """
    Dataset genérico para MTCNN (P-Net, R-Net, O-Net)
    
    Formato de anotação:
    image_path class x1 y1 x2 y2 landmark1_x landmark1_y ... landmark5_x landmark5_y
    
    Classes:
    0: negative
    1: positive  
    2: part
    3: landmark
    """
    
    def __init__(self, annotation_file, input_size, transform=None):
        """
        Args:
            annotation_file: path para arquivo de anotações
            input_size: tamanho de input (12 para P-Net, 24 para R-Net, 48 para O-Net)
            transform: transformações opcionais
        """
        self.input_size = input_size
        self.transform = transform
        
        # Carregar anotações
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) < 6:
                    continue
                
                sample = {
                    'img_path': line[0],
                    'class': int(line[1]),  # 0=neg, 1=pos, 2=part, 3=landmark
                    'bbox': np.array([float(x) for x in line[2:6]]),  # x1,y1,x2,y2
                    'landmarks': np.array([float(x) for x in line[6:]]) if len(line) > 6 else None
                }
                self.samples.append(sample)
        
        print(f"Carregadas {len(self.samples)} amostras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carregar imagem
        img = Image.open(sample['img_path']).convert('RGB')
        img = np.array(img)
        
        # Crop bbox
        x1, y1, x2, y2 = sample['bbox'].astype(np.int32)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img.shape[1] - 1, x2)
        y2 = min(img.shape[0] - 1, y2)
        
        img_crop = img[y1:y2+1, x1:x2+1]
        
        # Resize
        img_resized = cv2.resize(img_crop, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_LINEAR)
        
        # Normalizar para [-1, 1]
        img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)  # HWC -> CHW
        
        # Classe e tipo de sample
        cls_target = 1 if sample['class'] == 1 else 0  # Binary: face ou não
        sample_type = sample['class']
        
        # Bounding box target (offset normalizado)
        box_target = torch.zeros(4)
        if sample['class'] in [1, 2]:  # positive ou part
            # Offsets já estão no formato correto do arquivo
            box_target = torch.from_numpy(sample['bbox']).float()
        
        # Landmark target (normalizado)
        landmark_target = torch.zeros(10)
        if sample['class'] == 3 and sample['landmarks'] is not None:
            landmark_target = torch.from_numpy(sample['landmarks']).float()
        
        return {
            'image': img_tensor,
            'cls_target': torch.tensor(cls_target, dtype=torch.long),
            'box_target': box_target,
            'landmark_target': landmark_target,
            'sample_type': torch.tensor(sample_type, dtype=torch.long)
        }


def parse_wider_face_annotation(anno_file):
    """
    Parser para anotações do WIDER FACE
    
    Args:
        anno_file: path para arquivo wider_face_train.txt ou similar
    
    Returns:
        annotations: lista de dicts com image_path e bboxes
    """
    annotations = []
    
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Linha com nome da imagem
        img_path = lines[i].strip()
        i += 1
        
        # Número de faces
        num_faces = int(lines[i].strip())
        i += 1
        
        # Bboxes das faces
        bboxes = []
        for j in range(num_faces):
            bbox_info = lines[i].strip().split()
            i += 1
            
            # Formato: x y w h blur expression illumination invalid occlusion pose
            x, y, w, h = [int(float(v)) for v in bbox_info[:4]]
            
            # Ignorar faces inválidas
            if len(bbox_info) >= 8 and int(bbox_info[7]) == 1:
                continue
            
            # Converter para formato x1,y1,x2,y2
            bbox = [x, y, x + w - 1, y + h - 1]
            bboxes.append(bbox)
        
        if len(bboxes) > 0:
            annotations.append({
                'image_path': img_path,
                'bboxes': np.array(bboxes)
            })
    
    return annotations


def parse_celeba_landmarks(landmark_file):
    """
    Parser para landmarks do CelebA
    
    Args:
        landmark_file: path para list_landmarks_celeba.txt
    
    Returns:
        landmarks_dict: dict {image_name: landmarks}
    """
    landmarks_dict = {}
    
    with open(landmark_file, 'r') as f:
        # Pular header
        num_images = int(f.readline().strip())
        f.readline()  # Pular linha de nomes
        
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            
            # 5 landmarks: lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y ...
            landmarks = np.array([int(x) for x in parts[1:]], dtype=np.float32)
            landmarks_dict[img_name] = landmarks
    
    return landmarks_dict


def random_crop_with_bbox(img, bbox, crop_scale=(0.8, 1.2), shift=0.1):
    """
    Random crop mantendo bbox dentro da imagem
    
    Args:
        img: numpy array [H, W, C]
        bbox: numpy array [4] - (x1, y1, x2, y2)
        crop_scale: tuple - fator de escala do crop
        shift: float - máximo shift relativo
    
    Returns:
        cropped_img: imagem croppada
        new_bbox: bbox ajustada
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # Random scale
    scale = random.uniform(*crop_scale)
    new_w = int(bbox_w * scale)
    new_h = int(bbox_h * scale)
    
    # Random shift
    shift_x = random.uniform(-shift, shift) * bbox_w
    shift_y = random.uniform(-shift, shift) * bbox_h
    
    # Nova posição
    cx = (x1 + x2) / 2 + shift_x
    cy = (y1 + y2) / 2 + shift_y
    
    # Crop bounds
    crop_x1 = int(max(0, cx - new_w / 2))
    crop_y1 = int(max(0, cy - new_h / 2))
    crop_x2 = int(min(w, cx + new_w / 2))
    crop_y2 = int(min(h, cy + new_h / 2))
    
    # Crop
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Ajustar bbox
    new_bbox = np.array([
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1
    ])
    
    return cropped_img, new_bbox


def augment_image(img, prob=0.5):
    """
    Augmentação de imagem para treinamento
    
    Args:
        img: numpy array [H, W, C]
        prob: probabilidade de aplicar cada augmentação
    
    Returns:
        augmented_img: imagem aumentada
    """
    # Horizontal flip
    if random.random() < prob:
        img = cv2.flip(img, 1)
    
    # Brightness
    if random.random() < prob:
        beta = random.uniform(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
    
    # Contrast
    if random.random() < prob:
        alpha = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    
    # Gaussian blur
    if random.random() < prob * 0.3:  # Menos frequente
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img


if __name__ == "__main__":
    print("="*60)
    print("TESTE DE DATA UTILS")
    print("="*60)
    
    # Exemplo de uso
    print("\nExemplo de formato de anotação:")
    print("path/to/img.jpg 1 10 20 50 60 0.2 0.3 0.7 0.3 0.45 0.5 0.3 0.7 0.6 0.7")
    print("\nOnde:")
    print("- path/to/img.jpg: caminho da imagem")
    print("- 1: classe (0=neg, 1=pos, 2=part, 3=landmark)")
    print("- 10 20 50 60: bbox (x1, y1, x2, y2)")
    print("- 0.2 0.3 ... : landmarks normalizados (5 pontos x,y)")
    
    print("\n✓ Data utils prontos!")