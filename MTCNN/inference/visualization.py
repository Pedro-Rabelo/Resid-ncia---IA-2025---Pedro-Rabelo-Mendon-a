import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image, bboxes, scores=None, landmarks=None, 
                    thickness=2, font_scale=0.5):
    """
    Desenha detecções em uma imagem
    
    Args:
        image: numpy array [H, W, 3] RGB
        bboxes: numpy array [N, 4] - (x1, y1, x2, y2)
        scores: numpy array [N] - confidence scores (opcional)
        landmarks: numpy array [N, 10] - 5 pontos (x,y) (opcional)
        thickness: espessura das linhas
        font_scale: tamanho da fonte
    
    Returns:
        image_with_detections: imagem com detecções desenhadas
    """
    # Copiar para não modificar original
    img_draw = image.copy()
    
    # Converter para BGR para OpenCV
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    
    # Cores
    bbox_color = (0, 255, 0)  # Verde
    landmark_colors = [
        (255, 0, 0),    # Olho esquerdo - Vermelho
        (0, 0, 255),    # Olho direito - Azul
        (0, 255, 255),  # Nariz - Ciano
        (255, 255, 0),  # Boca esquerda - Amarelo
        (255, 0, 255)   # Boca direita - Magenta
    ]
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(np.int32)
        
        # Desenhar bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), bbox_color, thickness)
        
        # Desenhar score
        if scores is not None:
            score_text = f'{scores[i]:.2f}'
            cv2.putText(img_draw, score_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, bbox_color, thickness)
        
        # Desenhar landmarks
        if landmarks is not None:
            lmk = landmarks[i].reshape(5, 2).astype(np.int32)
            for j, (lx, ly) in enumerate(lmk):
                cv2.circle(img_draw, (lx, ly), radius=2, 
                          color=landmark_colors[j], thickness=-1)
    
    # Converter de volta para RGB
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    
    return img_draw


def draw_detections_matplotlib(image, bboxes, scores=None, landmarks=None, 
                               save_path=None, show=True):
    """
    Desenha detecções usando matplotlib (melhor para análise)
    
    Args:
        image: numpy array [H, W, 3] RGB
        bboxes: numpy array [N, 4]
        scores: numpy array [N] (opcional)
        landmarks: numpy array [N, 10] (opcional)
        save_path: path para salvar (opcional)
        show: se True, mostra a figura
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Cores para landmarks
    landmark_colors = ['red', 'blue', 'cyan', 'yellow', 'magenta']
    landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Bounding box
        rect = Rectangle((x1, y1), w, h, linewidth=2, 
                        edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        
        # Score
        if scores is not None:
            ax.text(x1, y1-5, f'{scores[i]:.3f}', 
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Landmarks
        if landmarks is not None:
            lmk = landmarks[i].reshape(5, 2)
            for j, (lx, ly) in enumerate(lmk):
                ax.plot(lx, ly, 'o', color=landmark_colors[j], 
                       markersize=6, markeredgecolor='white', markeredgewidth=1)
                
                # Label do landmark
                if i == 0:  # Apenas na primeira face
                    ax.text(lx+5, ly, landmark_names[j], 
                           color=landmark_colors[j], fontsize=8)
    
    ax.axis('off')
    ax.set_title(f'MTCNN Detection - {len(bboxes)} faces detected', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figura salva em: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_detection_grid(images, all_bboxes, all_scores=None, 
                         all_landmarks=None, grid_size=(2, 3)):
    """
    Cria grid de múltiplas detecções
    
    Args:
        images: lista de numpy arrays
        all_bboxes: lista de arrays de bboxes
        all_scores: lista de arrays de scores
        all_landmarks: lista de arrays de landmarks
        grid_size: tuple (rows, cols)
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (img, bboxes) in enumerate(zip(images, all_bboxes)):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        ax.imshow(img)
        
        scores = all_scores[idx] if all_scores else None
        landmarks = all_landmarks[idx] if all_landmarks else None
        
        # Desenhar detecções
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            rect = Rectangle((x1, y1), w, h, linewidth=2,
                           edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            
            if scores is not None:
                ax.text(x1, y1-5, f'{scores[i]:.2f}', 
                       color='green', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            if landmarks is not None:
                lmk = landmarks[i].reshape(5, 2)
                ax.plot(lmk[:, 0], lmk[:, 1], 'ro', markersize=3)
        
        ax.axis('off')
        ax.set_title(f'{len(bboxes)} faces', fontsize=10)
    
    # Esconder axes vazios
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_training_progress(checkpoint_dir, stage='pnet'):
    """
    Visualiza progresso do treinamento a partir dos logs do TensorBoard
    
    Args:
        checkpoint_dir: diretório com runs do tensorboard
        stage: 'pnet', 'rnet' ou 'onet'
    """
    from tensorboard.backend.event_processing import event_accumulator
    import os
    
    log_dir = os.path.join(checkpoint_dir, 'runs', stage)
    
    if not os.path.exists(log_dir):
        print(f"❌ Log dir não encontrado: {log_dir}")
        return
    
    # Carregar eventos
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Extrair métricas
    tags = ea.Tags()['scalars']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss total
    if 'Loss/total' in tags:
        events = ea.Scalars('Loss/total')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        axes[0, 0].plot(steps, values, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Losses individuais
    loss_tags = ['Loss/cls', 'Loss/box', 'Loss/landmark']
    colors = ['red', 'green', 'blue']
    
    for tag, color in zip(loss_tags, colors):
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            axes[0, 1].plot(steps, values, color=color, 
                           label=tag.split('/')[-1], linewidth=2)
    
    axes[0, 1].set_title('Individual Losses', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'Learning_rate' in tags:
        events = ea.Scalars('Learning_rate')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        axes[1, 0].plot(steps, values, 'm-', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Hard sample mining stats
    if 'Mining/hard_ratio' in tags:
        events = ea.Scalars('Mining/hard_ratio')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        axes[1, 1].plot(steps, values, 'c-', linewidth=2)
        axes[1, 1].set_title('Hard Sample Ratio', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{stage.upper()} Training Progress', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compare_detections(image, detections_list, labels, save_path=None):
    """
    Compara diferentes métodos de detecção lado a lado
    
    Args:
        image: numpy array [H, W, 3]
        detections_list: lista de (bboxes, scores, landmarks)
        labels: lista de strings com nomes dos métodos
        save_path: path para salvar
    """
    n_methods = len(detections_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (detections, label) in enumerate(zip(detections_list, labels)):
        bboxes, scores, landmarks = detections
        
        ax = axes[idx]
        img_draw = draw_detections(image.copy(), bboxes, scores, landmarks)
        ax.imshow(img_draw)
        ax.axis('off')
        ax.set_title(f'{label}\n{len(bboxes)} faces', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_face_crops(image, bboxes, landmarks=None, save_path=None):
    """
    Visualiza crops de faces detectadas
    
    Args:
        image: numpy array [H, W, 3]
        bboxes: numpy array [N, 4]
        landmarks: numpy array [N, 10] (opcional)
        save_path: path para salvar
    """
    n_faces = len(bboxes)
    
    if n_faces == 0:
        print("Nenhuma face detectada!")
        return
    
    cols = min(5, n_faces)
    rows = (n_faces + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    if n_faces == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, bbox in enumerate(bboxes):
        if idx >= len(axes):
            break
        
        x1, y1, x2, y2 = bbox.astype(np.int32)
        
        # Adicionar margem
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        # Crop
        face_crop = image[y1:y2, x1:x2]
        
        ax = axes[idx]
        ax.imshow(face_crop)
        
        # Desenhar landmarks se disponíveis
        if landmarks is not None:
            lmk = landmarks[idx].reshape(5, 2)
            # Ajustar coordenadas para o crop
            lmk_crop = lmk - np.array([x1, y1])
            ax.plot(lmk_crop[:, 0], lmk_crop[:, 1], 'ro', markersize=4)
        
        ax.axis('off')
        ax.set_title(f'Face {idx+1}', fontsize=10)
    
    # Esconder axes vazios
    for idx in range(n_faces, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("="*60)
    print("TESTE DE VISUALIZAÇÃO")
    print("="*60)
    
    # Criar imagem de teste
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Criar detecções simuladas
    bboxes = np.array([
        [100, 100, 200, 200],
        [300, 150, 400, 250],
        [150, 300, 250, 400]
    ], dtype=np.float32)
    
    scores = np.array([0.99, 0.95, 0.88])
    
    landmarks = np.array([
        [120, 140, 180, 140, 150, 170, 130, 190, 170, 190],  # Face 1
        [320, 180, 380, 180, 350, 200, 330, 220, 370, 220],  # Face 2
        [170, 330, 230, 330, 200, 350, 180, 370, 220, 370]   # Face 3
    ], dtype=np.float32)
    
    # Teste 1: Desenhar detecções
    print("\n[Test 1] Desenhando detecções...")
    img_with_det = draw_detections(img, bboxes, scores, landmarks)
    print(f"  ✓ Imagem com detecções: {img_with_det.shape}")
    
    # Teste 2: Matplotlib
    print("\n[Test 2] Visualização matplotlib...")
    fig = draw_detections_matplotlib(img, bboxes, scores, landmarks, show=False)
    print("  ✓ Figura criada")
    plt.close()
    
    # Teste 3: Crops
    print("\n[Test 3] Visualizando crops...")
    fig = visualize_face_crops(img, bboxes, landmarks, save_path=None)
    print("  ✓ Crops visualizados")
    plt.close()
    
    print("\n✓ Todas as funções de visualização funcionando!")