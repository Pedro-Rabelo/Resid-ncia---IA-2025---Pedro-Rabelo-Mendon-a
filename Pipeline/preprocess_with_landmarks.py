import os
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import argparse


def align_face(img, landmarks):
    """
    Alinha face usando os 5 landmarks do MTCNN/RetinaFace
    
    Args:
        img: imagem numpy array (RGB)
        landmarks: array [5, 2] com (x, y) de cada landmark
        
    Returns:
        aligned: imagem alinhada 112x112
        tform: matriz de transformação afim aplicada
    """
    # Template para face frontal normalizada (112x112)
    # Posições ideais dos landmarks em uma face alinhada
    template = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041]   # right mouth corner
    ], dtype=np.float32)
    
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Calcula transformação afim (escala, rotação, translação)
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    
    # Aplica transformação
    aligned = cv2.warpAffine(
        img, tform, (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return aligned, tform


def normalize_landmarks(landmarks, img_size=112):
    """
    Normaliza landmarks para [0, 1] para usar no treinamento
    
    Args:
        landmarks: array [5, 2] com coordenadas (x, y)
        img_size: tamanho da imagem (assumindo quadrada)
    
    Returns:
        landmarks_flat: lista [10] com [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
    """
    landmarks_normalized = np.array(landmarks, dtype=np.float32)
    landmarks_normalized[:, 0] /= img_size  # normaliza x
    landmarks_normalized[:, 1] /= img_size  # normaliza y
    return landmarks_normalized.flatten().tolist()


def validate_face(landmarks, bbox, min_size=20):
    """
    Valida se a face detectada é adequada para treinamento
    
    Args:
        landmarks: lista de 5 landmarks [[x,y], ...]
        bbox: bounding box [x, y, w, h]
        min_size: tamanho mínimo da face em pixels
        
    Returns:
        is_valid: bool indicando se é válida
        reason: string com motivo se inválida
    """
    x, y, w, h = bbox
    
    # 1. Verifica tamanho mínimo
    if w < min_size or h < min_size:
        return False, f"Face too small ({w}x{h})"
    
    # 2. Verifica se landmarks estão dentro do bbox
    for i, lm in enumerate(landmarks):
        if not (x <= lm[0] <= x+w and y <= lm[1] <= y+h):
            return False, f"Landmark {i} outside bbox"
    
    # 3. Verifica ordem vertical (olhos acima de nariz acima de boca)
    eyes_y = (landmarks[0][1] + landmarks[1][1]) / 2
    nose_y = landmarks[2][1]
    mouth_y = (landmarks[3][1] + landmarks[4][1]) / 2
    
    if not (eyes_y < nose_y < mouth_y):
        return False, "Invalid landmark vertical ordering"
    
    # 4. Verifica se olhos estão aproximadamente alinhados horizontalmente
    eye_diff = abs(landmarks[0][1] - landmarks[1][1])
    eye_distance = abs(landmarks[0][0] - landmarks[1][0])
    
    if eye_distance > 0 and eye_diff / eye_distance > 0.3:
        return False, "Eyes not horizontally aligned"
    
    return True, "Valid"


def preprocess_vggface2_dataset(input_root, output_root, output_landmarks_json, 
                                min_size=20, min_images_per_class=2):
    """
    Processa VGGFace2 dataset completo:
    1. Detecta faces com RetinaFace
    2. Valida detecções
    3. Alinha faces para 112x112
    4. Salva landmarks normalizados
    
    Args:
        input_root: diretório com imagens originais do VGGFace2
        output_root: diretório para imagens alinhadas 112x112
        output_landmarks_json: arquivo JSON para salvar landmarks
        min_size: tamanho mínimo da face em pixels
        min_images_per_class: mínimo de imagens por identidade (default: 2)
    
    Returns:
        landmarks_dict: dicionário com landmarks
        stats: estatísticas do processamento
    """
    
    # Importa detector de faces
    try:
        from uniface import RetinaFace
        import torch
        
        # Configura device
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("✓ Usando CPU")
        
        # Inicializa detector (uniface detecta GPU automaticamente)
        detector = RetinaFace()
        detector_name = "RetinaFace (uniface)"
        
    except ImportError:
        print("❌ RetinaFace não disponível!")
        print("Instale com: pip install uniface")
        raise ImportError("Detector de faces não disponível!")
    
    print(f"\n{'='*70}")
    print(f"PRÉ-PROCESSAMENTO VGGFACE2 COM {detector_name}")
    print(f"{'='*70}\n")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Landmarks JSON: {output_landmarks_json}")
    print(f"Min face size: {min_size}px")
    print(f"Min images/identity: {min_images_per_class}\n")
    
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dicionário para armazenar landmarks
    landmarks_dict = {}
    
    # Estatísticas de processamento
    stats = {
        "total_images": 0,
        "success": 0,
        "no_face": 0,
        "multiple_faces": 0,
        "invalid_landmarks": 0,
        "too_small": 0,
        "error": 0,
        "skipped_identities": 0
    }
    
    # Lista todas as identidades (diretórios)
    identity_folders = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    print(f"📊 Encontradas {len(identity_folders):,} identidades")
    print(f"🚀 Iniciando processamento...\n")
    
    # Processa cada identidade
    for identity_folder in tqdm(identity_folders, desc="Processando identidades"):
        identity_name = identity_folder.name
        output_identity_path = output_path / identity_name
        output_identity_path.mkdir(exist_ok=True)
        
        # Lista todas as imagens da identidade
        image_files = list(identity_folder.glob("*.jpg")) + \
                     list(identity_folder.glob("*.png")) + \
                     list(identity_folder.glob("*.jpeg")) + \
                     list(identity_folder.glob("*.JPG"))
        
        identity_success = 0
        
        for img_file in image_files:
            stats["total_images"] += 1
            
            try:
                # Carrega imagem
                img = cv2.imread(str(img_file))
                if img is None:
                    stats["error"] += 1
                    continue
                
                # Converte BGR para RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detecta faces
                result = detector.detect(img_rgb)
                
                if result is None or len(result) == 0:
                    stats["no_face"] += 1
                    continue
                
                # uniface retorna (detections, landmarks)
                # detections: [N, 5] = [x1, y1, x2, y2, score]
                # landmarks: [N, 5, 2]
                detections, keypoints_array = result
                
                if len(detections) == 0:
                    stats["no_face"] += 1
                    continue
                
                # Extrai boxes e scores
                boxes = detections[:, :4]  # [N, 4]
                scores = detections[:, 4]   # [N]
                
                if len(boxes) > 1:
                    stats["multiple_faces"] += 1
                    best_idx = np.argmax(scores)
                else:
                    best_idx = 0
                
                # Extrai box e landmarks da melhor face
                box = boxes[best_idx]
                score = scores[best_idx]
                keypoints_face = keypoints_array[best_idx]  # [5, 2]
                
                # Converte box para [x, y, w, h]
                x1, y1, x2, y2 = box
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                
                # Landmarks
                landmarks_array = keypoints_face.astype(np.float32)
                
                # Valida face
                is_valid, reason = validate_face(landmarks_array, bbox, min_size)
                if not is_valid:
                    if "too small" in reason.lower():
                        stats["too_small"] += 1
                    else:
                        stats["invalid_landmarks"] += 1
                    continue
                
                # Alinha face
                aligned_face, tform = align_face(img_rgb, landmarks_array)
                
                # Transforma landmarks para espaço alinhado
                landmarks_homogeneous = np.hstack([
                    landmarks_array,
                    np.ones((5, 1))
                ])
                landmarks_aligned = landmarks_homogeneous @ tform.T
                
                # Normaliza landmarks
                landmarks_normalized = normalize_landmarks(landmarks_aligned)
                
                # Salva imagem alinhada
                output_file = output_identity_path / img_file.name
                aligned_pil = Image.fromarray(aligned_face)
                aligned_pil.save(output_file, quality=95)
                
                # Salva landmarks
                key = f"{identity_name}/{img_file.name}"
                landmarks_dict[key] = landmarks_normalized
                
                stats["success"] += 1
                identity_success += 1
                
            except Exception as e:
                stats["error"] += 1
                if stats["error"] <= 10:
                    print(f"\n⚠️ Erro em {img_file.name}: {e}")
                continue
        
        # Remove diretório se nenhuma imagem foi processada com sucesso
        if identity_success == 0:
            try:
                output_identity_path.rmdir()
                stats["skipped_identities"] += 1
            except:
                pass
    
    # Filtra identidades com poucas imagens
    if min_images_per_class > 1:
        print(f"\n🔍 Filtrando identidades com menos de {min_images_per_class} imagens...")
        
        # Conta imagens por identidade
        identity_counts = {}
        for key in landmarks_dict.keys():
            identity = key.split('/')[0]
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
        
        # Remove identidades com poucas imagens
        filtered_landmarks = {}
        removed_identities = set()
        
        for key, landmarks in landmarks_dict.items():
            identity = key.split('/')[0]
            if identity_counts[identity] >= min_images_per_class:
                filtered_landmarks[key] = landmarks
            else:
                removed_identities.add(identity)
        
        # Remove diretórios físicos das identidades filtradas
        for identity in removed_identities:
            identity_path = output_path / identity
            if identity_path.exists():
                for img_file in identity_path.iterdir():
                    img_file.unlink()
                identity_path.rmdir()
        
        print(f"✓ Removidas {len(removed_identities)} identidades com < {min_images_per_class} imagens")
        landmarks_dict = filtered_landmarks
    
    # Salva landmarks em JSON
    print(f"\n💾 Salvando landmarks...")
    with open(output_landmarks_json, 'w') as f:
        json.dump(landmarks_dict, f, indent=2)
    
    # Salva estatísticas
    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Relatório final
    print(f"\n{'='*70}")
    print("RESUMO DO PRÉ-PROCESSAMENTO")
    print(f"{'='*70}")
    print(f"Total de imagens processadas: {stats['total_images']:,}")
    print(f"✅ Sucesso:                    {stats['success']:,}")
    print(f"❌ Sem face detectada:         {stats['no_face']:,}")
    print(f"⚠️  Múltiplas faces:            {stats['multiple_faces']:,}")
    print(f"⚠️  Landmarks inválidos:        {stats['invalid_landmarks']:,}")
    print(f"⚠️  Face muito pequena:         {stats['too_small']:,}")
    print(f"⚠️  Erros:                      {stats['error']:,}")
    print(f"📁 Identidades removidas:      {stats['skipped_identities']:,}")
    print(f"{'='*70}")
    
    success_rate = (stats['success'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
    print(f"\n📊 Taxa de sucesso: {success_rate:.2f}%")
    print(f"✅ Imagens alinhadas: {output_root}")
    print(f"✅ Landmarks salvos: {output_landmarks_json}")
    print(f"✅ Estatísticas: {stats_file}\n")
    
    return landmarks_dict, stats


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Pré-processamento VGGFace2: detecção, alinhamento e extração de landmarks"
    )
    
    parser.add_argument(
        '--input-root',
        type=str,
        default='data/raw/vggface2_112x112',
        help='Diretório com imagens originais do VGGFace2 (default: data/raw/vggface2_112x112)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='data/train/vggface2_aligned_112x112',
        help='Diretório para salvar imagens alinhadas (default: data/train/vggface2_aligned_112x112)'
    )
    parser.add_argument(
        '--landmarks-json',
        type=str,
        default='data/train/vggface2_landmarks.json',
        help='Arquivo JSON para salvar landmarks (default: data/train/vggface2_landmarks.json)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=20,
        help='Tamanho mínimo da face em pixels (default: 20)'
    )
    parser.add_argument(
        '--min-images-per-class',
        type=int,
        default=2,
        help='Número mínimo de imagens por identidade (default: 2)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*70)
    print("VGGFACE2 - FACE DETECTION AND ALIGNMENT PIPELINE")
    print("="*70)
    
    landmarks_dict, stats = preprocess_vggface2_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        output_landmarks_json=args.landmarks_json,
        min_size=args.min_size,
        min_images_per_class=args.min_images_per_class
    )
    
    if stats['success'] > 0:
        print("✅ Pré-processamento concluído com sucesso!")
        print("\nPróximos passos:")
        print(f"  1. Verificar dataset:")
        print(f"     python verify_dataset_structure.py --root {args.output_root}")
        print(f"\n  2. Treinar modelo:")
        print(f"     python train_multitask.py \\")
        print(f"       --root {args.output_root} \\")
        print(f"       --landmarks-json {args.landmarks_json}")
    else:
        print("❌ Nenhuma imagem foi processada com sucesso!")
        print("\nVerifique:")
        print("  - Caminho do input está correto")
        print("  - RetinaFace está instalado (pip install uniface)")
        print("  - Imagens são válidas")