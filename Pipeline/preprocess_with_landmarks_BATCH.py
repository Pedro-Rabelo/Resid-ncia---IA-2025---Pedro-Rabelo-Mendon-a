import os
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import argparse
from collections import defaultdict


def align_face(img, landmarks):
    """Alinha face usando os 5 landmarks"""
    template = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041]   # right mouth
    ], dtype=np.float32)
    
    landmarks = np.array(landmarks, dtype=np.float32)
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    aligned = cv2.warpAffine(img, tform, (112, 112))
    
    return aligned, tform


def normalize_landmarks(landmarks, img_size=112):
    """Normaliza landmarks para [0, 1]"""
    landmarks_normalized = np.array(landmarks, dtype=np.float32)
    landmarks_normalized[:, 0] /= img_size
    landmarks_normalized[:, 1] /= img_size
    return landmarks_normalized.flatten().tolist()


def validate_face(landmarks, bbox, min_size=20):
    """Valida face detectada"""
    x, y, w, h = bbox
    
    if w < min_size or h < min_size:
        return False, f"Face too small ({w}x{h})"
    
    for i, lm in enumerate(landmarks):
        if not (x <= lm[0] <= x+w and y <= lm[1] <= y+h):
            return False, f"Landmark {i} outside bbox"
    
    eyes_y = (landmarks[0][1] + landmarks[1][1]) / 2
    nose_y = landmarks[2][1]
    mouth_y = (landmarks[3][1] + landmarks[4][1]) / 2
    
    if not (eyes_y < nose_y < mouth_y):
        return False, "Invalid landmark vertical ordering"
    
    return True, "Valid"


def process_batch(detector, image_batch, image_paths, output_path, identity_name, 
                  landmarks_dict, stats, min_size=20):
    """
    Processa um batch de imagens de uma vez
    
    Args:
        detector: RetinaFace detector
        image_batch: lista de imagens RGB
        image_paths: lista de paths das imagens
        output_path: diretório de saída
        identity_name: nome da identidade
        landmarks_dict: dicionário para salvar landmarks
        stats: dicionário de estatísticas
        min_size: tamanho mínimo da face
    
    Returns:
        número de imagens processadas com sucesso
    """
    if len(image_batch) == 0:
        return 0
    
    success_count = 0
    output_identity_path = output_path / identity_name
    output_identity_path.mkdir(exist_ok=True)
    
    try:
        # Detecta faces em batch (muito mais rápido!)
        batch_results = []
        for img in image_batch:
            result = detector.detect(img)
            batch_results.append(result)
        
        # Processa cada resultado
        for idx, (result, img_rgb, img_path) in enumerate(zip(batch_results, image_batch, image_paths)):
            stats["total_images"] += 1
            
            try:
                if result is None or len(result) == 0:
                    stats["no_face"] += 1
                    continue
                
                detections, keypoints_array = result
                
                if len(detections) == 0:
                    stats["no_face"] += 1
                    continue
                
                boxes = detections[:, :4]
                scores = detections[:, 4]
                
                if len(boxes) > 1:
                    stats["multiple_faces"] += 1
                    best_idx = np.argmax(scores)
                else:
                    best_idx = 0
                
                box = boxes[best_idx]
                keypoints_face = keypoints_array[best_idx]
                
                x1, y1, x2, y2 = box
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                
                landmarks_array = keypoints_face.astype(np.float32)
                
                is_valid, reason = validate_face(landmarks_array, bbox, min_size)
                if not is_valid:
                    if "too small" in reason.lower():
                        stats["too_small"] += 1
                    else:
                        stats["invalid_landmarks"] += 1
                    continue
                
                aligned_face, tform = align_face(img_rgb, landmarks_array)
                
                landmarks_homogeneous = np.hstack([
                    landmarks_array,
                    np.ones((5, 1))
                ])
                landmarks_aligned = landmarks_homogeneous @ tform.T
                landmarks_normalized = normalize_landmarks(landmarks_aligned)
                
                output_file = output_identity_path / img_path.name
                aligned_pil = Image.fromarray(aligned_face)
                aligned_pil.save(output_file, quality=95)
                
                key = f"{identity_name}/{img_path.name}"
                landmarks_dict[key] = landmarks_normalized
                
                stats["success"] += 1
                success_count += 1
                
            except Exception as e:
                stats["error"] += 1
                if stats["error"] <= 10:
                    print(f"\n⚠️ Erro em {img_path.name}: {e}")
                continue
        
    except Exception as e:
        print(f"\n❌ Erro no batch: {e}")
        stats["error"] += len(image_batch)
    
    return success_count


def preprocess_vggface2_dataset_batch(input_root, output_root, output_landmarks_json, 
                                      min_size=20, min_images_per_class=2, batch_size=16):
    """
    Processa VGGFace2 com batch processing
    
    Args:
        batch_size: número de imagens a processar simultaneamente (default: 16)
                   - RTX 3050 6GB: use 8-16
                   - RTX 3060 12GB: use 16-32
                   - RTX 3090 24GB: use 32-64
    """
    
    try:
        from uniface import RetinaFace
        import torch
        
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = 'cpu'
            print("✓ Usando CPU")
        
        detector = RetinaFace()
        detector_name = "RetinaFace (uniface) - BATCH MODE"
        
    except ImportError:
        print("❌ RetinaFace não disponível!")
        raise ImportError("Detector de faces não disponível!")
    
    print(f"\n{'='*70}")
    print(f"PRÉ-PROCESSAMENTO VGGFACE2 COM {detector_name}")
    print(f"{'='*70}\n")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Landmarks JSON: {output_landmarks_json}")
    print(f"Min face size: {min_size}px")
    print(f"Min images/identity: {min_images_per_class}")
    print(f"🚀 Batch size: {batch_size} (OTIMIZADO PARA GPU!)\n")
    
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    landmarks_dict = {}
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
    
    identity_folders = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    print(f"📊 Encontradas {len(identity_folders):,} identidades")
    print(f"🚀 Iniciando processamento em BATCH...\n")
    
    for identity_folder in tqdm(identity_folders, desc="Processando identidades"):
        identity_name = identity_folder.name
        
        image_files = list(identity_folder.glob("*.jpg")) + \
                     list(identity_folder.glob("*.png")) + \
                     list(identity_folder.glob("*.jpeg"))
        
        # Processa em batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            
            # Carrega batch de imagens
            image_batch = []
            valid_paths = []
            
            for img_file in batch_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        image_batch.append(img_rgb)
                        valid_paths.append(img_file)
                except:
                    stats["error"] += 1
                    continue
            
            # Processa batch
            if image_batch:
                process_batch(
                    detector, image_batch, valid_paths,
                    output_path, identity_name,
                    landmarks_dict, stats, min_size
                )
    
    # Filtra identidades com poucas imagens
    if min_images_per_class > 1:
        print(f"\n🔍 Filtrando identidades com < {min_images_per_class} imagens...")
        
        identity_counts = defaultdict(int)
        for key in landmarks_dict.keys():
            identity = key.split('/')[0]
            identity_counts[identity] += 1
        
        filtered_landmarks = {}
        removed_identities = set()
        
        for key, landmarks in landmarks_dict.items():
            identity = key.split('/')[0]
            if identity_counts[identity] >= min_images_per_class:
                filtered_landmarks[key] = landmarks
            else:
                removed_identities.add(identity)
        
        for identity in removed_identities:
            identity_path = output_path / identity
            if identity_path.exists():
                for img_file in identity_path.iterdir():
                    img_file.unlink()
                identity_path.rmdir()
        
        print(f"✓ Removidas {len(removed_identities)} identidades")
        landmarks_dict = filtered_landmarks
    
    # Salva landmarks
    print(f"\n💾 Salvando landmarks...")
    with open(output_landmarks_json, 'w') as f:
        json.dump(landmarks_dict, f, indent=2)
    
    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Relatório
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
    print(f"{'='*70}")
    
    success_rate = (stats['success'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
    print(f"\n📊 Taxa de sucesso: {success_rate:.2f}%")
    print(f"✅ Landmarks salvos: {output_landmarks_json}\n")
    
    return landmarks_dict, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Pré-processamento VGGFace2 com BATCH")
    
    parser.add_argument('--input-root', type=str, default='data/raw/vggface2_112x112')
    parser.add_argument('--output-root', type=str, default='data/train/vggface2_aligned_112x112')
    parser.add_argument('--landmarks-json', type=str, default='data/train/vggface2_landmarks.json')
    parser.add_argument('--min-size', type=int, default=20)
    parser.add_argument('--min-images-per-class', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size para processamento GPU (RTX 3050: 8-16, RTX 3060: 16-32)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*70)
    print("VGGFACE2 - BATCH FACE DETECTION PIPELINE (GPU OPTIMIZED)")
    print("="*70)
    
    landmarks_dict, stats = preprocess_vggface2_dataset_batch(
        input_root=args.input_root,
        output_root=args.output_root,
        output_landmarks_json=args.landmarks_json,
        min_size=args.min_size,
        min_images_per_class=args.min_images_per_class,
        batch_size=args.batch_size
    )
    
    if stats['success'] > 0:
        print("✅ Pré-processamento concluído!")
    else:
        print("❌ Falha no pré-processamento!")