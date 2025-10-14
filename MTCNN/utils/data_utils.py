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
    Parser OTIMIZADO para anotações do WIDER FACE
    
    Args:
        anno_file: path para wider_face_train_bbx_gt.txt
    
    Returns:
        annotations: lista de dicts com image_path e bboxes
        
    Formato WIDER FACE (sem linhas vazias):
        image_path.jpg
        num_faces
        x1 y1 w h blur expression illumination invalid occlusion pose
        x1 y1 w h blur expression illumination invalid occlusion pose
        ...
        next_image_path.jpg
        num_faces
        ...
    """
    import numpy as np
    
    annotations = []
    
    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    total_lines = len(lines)
    skipped_images = 0
    invalid_boxes = 0
    
    while i < total_lines:
        # ============ LINHA 1: IMAGEM ============
        img_path = lines[i].strip()
        
        # Validar se é realmente um caminho de imagem
        if not img_path or not img_path.endswith('.jpg'):
            print(f"⚠️ Linha {i}: Esperado caminho de imagem, encontrado: '{img_path}'")
            skipped_images += 1
            i += 1
            continue
        
        i += 1
        
        # Verificar se não chegou ao fim do arquivo
        if i >= total_lines:
            print(f"⚠️ Arquivo terminou após imagem: {img_path}")
            break
        
        # ============ LINHA 2: NÚMERO DE FACES ============
        num_faces_line = lines[i].strip()
        
        try:
            num_faces = int(num_faces_line)
        except ValueError:
            print(f"⚠️ Linha {i}: Esperado número de faces, encontrado: '{num_faces_line}'")
            print(f"   Imagem: {img_path}")
            skipped_images += 1
            i += 1
            continue
        
        i += 1
        
        # ============ LINHAS 3+: BOUNDING BOXES ============
        bboxes = []
        
        # Caso especial: imagens com 0 faces
        if num_faces == 0:
            # WIDER FACE coloca uma linha placeholder para 0 faces: "0 0 0 0 0 0 0 0 0 0"
            # Precisamos pular esta linha
            if i < total_lines:
                placeholder_line = lines[i].strip()
                # Verificar se é a linha placeholder (não é um caminho de imagem)
                if not placeholder_line.endswith('.jpg'):
                    i += 1
            # Não adicionar imagens sem faces ao dataset
            continue
        
        # Processar cada bounding box
        for j in range(num_faces):
            if i >= total_lines:
                print(f"⚠️ Arquivo terminou ao processar bboxes de: {img_path}")
                print(f"   Esperadas {num_faces} faces, encontradas {j}")
                break
            
            bbox_line = lines[i].strip()
            
            # Validação de segurança: não deve ser um caminho de imagem
            if bbox_line.endswith('.jpg'):
                print(f"⚠️ Linha {i}: Encontrado caminho de imagem ao processar bboxes")
                print(f"   Imagem atual: {img_path}")
                print(f"   Esperadas {num_faces} faces, processadas {j}")
                # Não incrementar i, deixar para próxima iteração do while
                break
            
            # Parse da bbox
            bbox_parts = bbox_line.split()
            
            # WIDER FACE formato: x y w h blur expression illumination invalid occlusion pose
            if len(bbox_parts) < 4:
                print(f"⚠️ Linha {i}: Bbox incompleta - {bbox_line}")
                i += 1
                invalid_boxes += 1
                continue
            
            try:
                x = int(bbox_parts[0])
                y = int(bbox_parts[1])
                w = int(bbox_parts[2])
                h = int(bbox_parts[3])
                
                # Converter para formato [x1, y1, x2, y2]
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                
                # Validar dimensões da bbox
                if w > 0 and h > 0:
                    bboxes.append([x1, y1, x2, y2])
                else:
                    invalid_boxes += 1
                
            except (ValueError, IndexError) as e:
                print(f"⚠️ Linha {i}: Erro ao converter bbox - {bbox_line}")
                print(f"   Erro: {e}")
                invalid_boxes += 1
            
            i += 1
        
        # Adicionar anotação somente se houver bboxes válidas
        if len(bboxes) > 0:
            annotations.append({
                'image_path': img_path,
                'bboxes': np.array(bboxes, dtype=np.float32)
            })
    
    # Relatório final
    print(f"\n{'='*70}")
    print(f"RELATÓRIO DO PARSER")
    print(f"{'='*70}")
    print(f"✓ Total de linhas processadas: {total_lines}")
    print(f"✓ Imagens válidas: {len(annotations)}")
    print(f"✓ Total de faces: {sum(len(a['bboxes']) for a in annotations)}")
    
    if skipped_images > 0:
        print(f"⚠️ Imagens puladas: {skipped_images}")
    if invalid_boxes > 0:
        print(f"⚠️ Bboxes inválidas: {invalid_boxes}")
    
    print(f"{'='*70}\n")
    
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