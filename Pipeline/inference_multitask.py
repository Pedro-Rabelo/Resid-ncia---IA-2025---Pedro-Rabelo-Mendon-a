import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from pathlib import Path

from models.mobilenetv3_multitask import mobilenetv3_large_multitask
from uniface import RetinaFace


def load_multitask_model(checkpoint_path, device):
    """
    Carrega modelo VGGFace2 treinado com multi-task learning
    
    Args:
        checkpoint_path: caminho para o checkpoint (.ckpt)
        device: torch device (cpu ou cuda)
    
    Returns:
        model: modelo carregado em modo de avaliação
    """
    print(f"📂 Carregando modelo de: {checkpoint_path}")
    
    # Cria modelo
    model = mobilenetv3_large_multitask(embedding_dim=512).to(device)
    
    # Carrega checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Carrega pesos
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Informações do checkpoint
    if 'epoch' in checkpoint:
        print(f"✓ Modelo do epoch {checkpoint['epoch']}")
    if 'best_lfw_accuracy' in checkpoint:
        print(f"✓ LFW accuracy: {checkpoint['best_lfw_accuracy']:.4f}")
    if 'num_classes' in checkpoint:
        print(f"✓ Treinado com {checkpoint['num_classes']:,} classes")
    
    print()
    
    return model


def align_face(img, landmarks):
    """
    Alinha face usando os 5 landmarks
    
    Args:
        img: imagem numpy array (RGB)
        landmarks: lista de 5 landmarks [[x,y], ...]
    
    Returns:
        aligned: imagem alinhada 112x112
    """
    # Template padrão para face frontal
    template = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041]   # right mouth
    ], dtype=np.float32)
    
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Calcula transformação afim
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    
    # Aplica transformação
    aligned = cv2.warpAffine(img, tform, (112, 112))
    
    return aligned


def extract_embedding(model, img_path, detector, device, verbose=False):
    """
    Extrai embedding de uma imagem
    
    Args:
        model: modelo carregado
        img_path: caminho para a imagem
        detector: detector de faces (RetinaFace)
        device: torch device
        verbose: se True, imprime informações
    
    Returns:
        embedding: vetor de features (512D)
        status: mensagem de status
    """
    if verbose:
        print(f"  Processando: {img_path}")
    
    # Carrega imagem
    if not os.path.exists(img_path):
        return None, f"File not found: {img_path}"
    
    img = cv2.imread(img_path)
    if img is None:
        return None, "Failed to load image"
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detecta faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        return None, "No face detected"
    
    # Usa a maior face se houver múltiplas
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
        if verbose:
            print(f"  ⚠️  {len(faces)} faces detectadas, usando a maior")
    
    face = faces[0]
    keypoints = face['keypoints']
    
    # Extrai landmarks
    landmarks = [
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ]
    
    # Alinha face
    aligned = align_face(img_rgb, landmarks)
    
    # Transforma para tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(device)
    
    # Extrai embedding (sem landmarks na inferência)
    with torch.no_grad():
        embedding = model.extract_features(img_tensor)
    
    if verbose:
        print(f"  ✓ Embedding extraído: {embedding.shape}")
    
    return embedding.cpu().numpy(), "Success"


def compute_similarity(embedding1, embedding2):
    """
    Calcula similaridade coseno entre dois embeddings
    
    Args:
        embedding1: primeiro embedding
        embedding2: segundo embedding
    
    Returns:
        similarity: similaridade no intervalo [-1, 1]
    """
    dot_product = np.dot(embedding1, embedding2.T)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2 + 1e-8)
    return similarity[0][0]


def compare_two_faces(model, detector, device, img1_path, img2_path, threshold=0.35):
    """
    Compara duas faces e determina se são da mesma pessoa
    
    Args:
        model: modelo carregado
        detector: detector de faces
        device: torch device
        img1_path: caminho para primeira imagem
        img2_path: caminho para segunda imagem
        threshold: limiar de similaridade (default: 0.35)
    
    Returns:
        similarity: valor de similaridade
        is_same: True se mesma pessoa, False caso contrário
    """
    print("="*60)
    print("COMPARAÇÃO DE FACES")
    print("="*60)
    
    # Extrai embeddings
    print(f"\n📸 Imagem 1: {img1_path}")
    emb1, status1 = extract_embedding(model, img1_path, detector, device, verbose=True)
    
    if emb1 is None:
        print(f"❌ Erro: {status1}")
        return None, None
    
    print(f"\n📸 Imagem 2: {img2_path}")
    emb2, status2 = extract_embedding(model, img2_path, detector, device, verbose=True)
    
    if emb2 is None:
        print(f"❌ Erro: {status2}")
        return None, None
    
    # Calcula similaridade
    similarity = compute_similarity(emb1, emb2)
    is_same = similarity > threshold
    
    print(f"\n{'='*60}")
    print(f"RESULTADO")
    print(f"{'='*60}")
    print(f"Similaridade:  {similarity:.4f}")
    print(f"Threshold:     {threshold:.4f}")
    print(f"Mesma pessoa:  {'✅ SIM' if is_same else '❌ NÃO'}")
    print(f"{'='*60}\n")
    
    return similarity, is_same


def batch_extract_embeddings(model, detector, device, image_folder, output_file=None):
    """
    Extrai embeddings de múltiplas imagens em uma pasta
    
    Args:
        model: modelo carregado
        detector: detector de faces
        device: torch device
        image_folder: pasta com imagens
        output_file: arquivo para salvar embeddings (opcional)
    
    Returns:
        embeddings_dict: dicionário {filename: embedding}
    """
    print("="*60)
    print(f"EXTRAÇÃO EM LOTE DE EMBEDDINGS")
    print("="*60)
    print(f"Pasta: {image_folder}\n")
    
    image_folder = Path(image_folder)
    
    # Lista imagens
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
    
    print(f"📊 Encontradas {len(image_files)} imagens\n")
    
    embeddings_dict = {}
    success_count = 0
    
    for img_path in image_files:
        embedding, status = extract_embedding(
            model, str(img_path), detector, device, verbose=False
        )
        
        if embedding is not None:
            embeddings_dict[img_path.name] = embedding.tolist()
            success_count += 1
            print(f"✓ {img_path.name}")
        else:
            print(f"✗ {img_path.name} - {status}")
    
    print(f"\n{'='*60}")
    print(f"Sucesso: {success_count}/{len(image_files)}")
    print(f"{'='*60}\n")
    
    # Salva em arquivo se especificado
    if output_file and embeddings_dict:
        import json
        with open(output_file, 'w') as f:
            json.dump(embeddings_dict, f, indent=2)
        print(f"✅ Embeddings salvos em: {output_file}\n")
    
    return embeddings_dict


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description="Inferência com modelo VGGFace2 treinado (multi-task learning)"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['compare', 'extract', 'batch'],
        default='compare',
        help='Modo de operação: compare (2 imagens), extract (1 imagem), batch (pasta)'
    )
    
    # Paths
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='weights/vggface2/mobilenetv3_vggface2_multitask_best.ckpt',
        help='Caminho para o checkpoint do modelo'
    )
    parser.add_argument(
        '--img1',
        type=str,
        help='Caminho para primeira imagem (mode=compare ou extract)'
    )
    parser.add_argument(
        '--img2',
        type=str,
        help='Caminho para segunda imagem (mode=compare)'
    )
    parser.add_argument(
        '--folder',
        type=str,
        help='Pasta com imagens (mode=batch)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Arquivo de saída para embeddings (mode=batch)'
    )
    
    # Parâmetros
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.35,
        help='Threshold de similaridade (default: 0.35)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID (-1 para CPU, default: 0)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    print(f"\n🖥️  Device: {device}\n")
    
    # Carrega modelo
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint não encontrado: {args.checkpoint}")
        return
    
    model = load_multitask_model(args.checkpoint, device)
    
    # Inicializa detector
    print("📷 Inicializando RetinaFace...")
    detector = RetinaFace(gpu_id=args.gpu if args.gpu >= 0 else -1)
    print("✓ Detector pronto\n")
    
    # Executa modo selecionado
    if args.mode == 'compare':
        if not args.img1 or not args.img2:
            print("❌ Mode 'compare' requer --img1 e --img2")
            return
        
        compare_two_faces(
            model, detector, device,
            args.img1, args.img2,
            threshold=args.threshold
        )
    
    elif args.mode == 'extract':
        if not args.img1:
            print("❌ Mode 'extract' requer --img1")
            return
        
        print(f"📸 Extraindo embedding de: {args.img1}\n")
        embedding, status = extract_embedding(
            model, args.img1, detector, device, verbose=True
        )
        
        if embedding is not None:
            print(f"\n✅ Embedding extraído com sucesso!")
            print(f"Shape: {embedding.shape}")
            print(f"Norma L2: {np.linalg.norm(embedding):.4f}")
        else:
            print(f"\n❌ Falha: {status}")
    
    elif args.mode == 'batch':
        if not args.folder:
            print("❌ Mode 'batch' requer --folder")
            return
        
        batch_extract_embeddings(
            model, detector, device,
            args.folder,
            output_file=args.output
        )


if __name__ == "__main__":
    main()