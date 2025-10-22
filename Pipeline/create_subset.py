import shutil
from pathlib import Path
import random
import argparse


def create_subset(input_root, output_root, n_identities=200, min_images=5, seed=42):
    """
    Cria subset do VGGFace2 selecionando identidades aleatórias
    
    Args:
        input_root: diretório com dataset VGGFace2 completo
        output_root: diretório para salvar subset
        n_identities: número de identidades a selecionar
        min_images: número mínimo de imagens por identidade
        seed: seed para reprodutibilidade
    """
    random.seed(seed)
    
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        print(f"❌ Erro: Diretório de entrada não encontrado: {input_root}")
        return
    
    # Remove output se já existir
    if output_path.exists():
        print(f"⚠️  Diretório {output_root} já existe. Removendo...")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CRIANDO SUBSET DO VGGFACE2")
    print("="*70)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Target: {n_identities} identidades")
    print(f"Min images/identity: {min_images}")
    print("="*70 + "\n")
    
    # Lista todas as identidades
    print("📊 Listando identidades...")
    identities = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"   Total encontrado: {len(identities):,}")
    
    # Filtra identidades com suficientes imagens
    print(f"\n🔍 Filtrando identidades com >= {min_images} imagens...")
    valid_identities = []
    for identity in identities:
        images = list(identity.glob('*.jpg')) + \
                 list(identity.glob('*.png')) + \
                 list(identity.glob('*.jpeg'))
        
        if len(images) >= min_images:
            valid_identities.append((identity, len(images)))
    
    print(f"   Identidades válidas: {len(valid_identities):,}/{len(identities):,}")
    
    if len(valid_identities) < n_identities:
        print(f"⚠️  Aviso: Apenas {len(valid_identities)} identidades disponíveis")
        n_identities = len(valid_identities)
    
    # Seleciona aleatoriamente
    print(f"\n🎲 Selecionando {n_identities} identidades aleatoriamente...")
    selected = random.sample(valid_identities, n_identities)
    
    # Copia identidades selecionadas
    print(f"\n📁 Copiando identidades...")
    total_images = 0
    
    for i, (identity, n_images) in enumerate(selected, 1):
        dest = output_path / identity.name
        shutil.copytree(identity, dest)
        total_images += n_images
        
        if i % 20 == 0 or i == len(selected):
            print(f"   Progresso: {i}/{len(selected)} identidades copiadas")
    
    print("\n" + "="*70)
    print("RESUMO")
    print("="*70)
    print(f"✅ Identidades copiadas: {len(selected):,}")
    print(f"✅ Total de imagens:     {total_images:,}")
    print(f"✅ Média de imagens/ID:  {total_images/len(selected):.1f}")
    print(f"✅ Subset salvo em:      {output_root}")
    print("="*70 + "\n")
    
    print("Próximo passo:")
    print(f"  python preprocess_with_landmarks.py \\")
    print(f"    --input-root {output_root} \\")
    print(f"    --output-root data/train/vggface2_aligned_subset \\")
    print(f"    --landmarks-json data/train/vggface2_landmarks_subset.json\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Criar subset do VGGFace2')
    
    parser.add_argument(
        '--input-root',
        type=str,
        default='data/raw/vggface2_112x112',
        help='Diretório com VGGFace2 completo (default: data/raw/vggface2_112x112)'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='data/raw/vggface2_subset_200',
        help='Diretório para salvar subset (default: data/raw/vggface2_subset_200)'
    )
    parser.add_argument(
        '--n-identities',
        type=int,
        default=200,
        help='Número de identidades a selecionar (default: 200)'
    )
    parser.add_argument(
        '--min-images',
        type=int,
        default=5,
        help='Número mínimo de imagens por identidade (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (default: 42)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    create_subset(
        input_root=args.input_root,
        output_root=args.output_root,
        n_identities=args.n_identities,
        min_images=args.min_images,
        seed=args.seed
    )