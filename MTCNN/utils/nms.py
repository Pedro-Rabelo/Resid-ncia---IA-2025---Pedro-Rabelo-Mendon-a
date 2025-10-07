import numpy as np


def nms(boxes, scores, threshold=0.5, mode='union'):
    """
    Non-Maximum Suppression (NMS)
    
    Suprime bounding boxes sobrepostas mantendo apenas as de maior score.
    
    Args:
        boxes: numpy array [N, 4] formato (x1, y1, x2, y2)
        scores: numpy array [N] scores de confiança
        threshold: float, IoU threshold para supressão
        mode: 'union' ou 'min' para cálculo de IoU
    
    Returns:
        keep: numpy array com índices das boxes mantidas
    
    Exemplo:
        >>> boxes = np.array([[100, 100, 200, 200], [110, 110, 210, 210]])
        >>> scores = np.array([0.9, 0.8])
        >>> keep = nms(boxes, scores, threshold=0.5)
        >>> kept_boxes = boxes[keep]
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    
    # Extrair coordenadas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calcular áreas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Ordenar por score decrescente
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # Pegar box com maior score
        i = order[0]
        keep.append(i)
        
        # Calcular IoU com boxes restantes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Área de interseção
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # Calcular IoU
        if mode == 'min':
            # IoU = inter / min(area1, area2)
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:  # union
            # IoU = inter / (area1 + area2 - inter)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Manter apenas boxes com IoU <= threshold
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def soft_nms(boxes, scores, threshold=0.5, sigma=0.5, score_threshold=0.001, mode='union'):
    """
    Soft-NMS: melhora o NMS tradicional ao decair scores ao invés de suprimir completamente
    
    Referência: Soft-NMS -- Improving Object Detection With One Line of Code
    
    Args:
        boxes: numpy array [N, 4]
        scores: numpy array [N]
        threshold: IoU threshold
        sigma: parâmetro do gaussiano para decay
        score_threshold: threshold mínimo de score
        mode: 'union' ou 'min'
    
    Returns:
        keep: índices das boxes mantidas
        new_scores: scores atualizados
    """
    boxes = boxes.copy()
    scores = scores.copy()
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    
    while len(scores) > 0:
        # Pegar box com maior score
        idx = scores.argmax()
        i = idx
        keep.append(i)
        
        if len(scores) == 1:
            break
        
        # Calcular IoU
        xx1 = np.maximum(x1[i], np.delete(x1, i))
        yy1 = np.maximum(y1[i], np.delete(y1, i))
        xx2 = np.minimum(x2[i], np.delete(x2, i))
        yy2 = np.minimum(y2[i], np.delete(y2, i))
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if mode == 'min':
            ovr = inter / np.minimum(areas[i], np.delete(areas, i))
        else:
            ovr = inter / (areas[i] + np.delete(areas, i) - inter)
        
        # Soft-NMS: decair scores baseado em IoU
        weight = np.exp(-(ovr * ovr) / sigma)
        
        # Atualizar scores
        scores_new = np.delete(scores, i)
        scores_new = scores_new * weight
        
        # Remover boxes com score muito baixo
        valid_idx = scores_new > score_threshold
        
        boxes = np.delete(boxes, i, axis=0)[valid_idx]
        scores = scores_new[valid_idx]
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    return np.array(keep, dtype=np.int32), scores


def nms_fast(boxes, scores, threshold=0.5):
    """
    Versão otimizada do NMS usando operações vetorizadas
    
    Args:
        boxes: numpy array [N, 4]
        scores: numpy array [N]
        threshold: IoU threshold
    
    Returns:
        keep: índices mantidos
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        # Vetorizar cálculo de IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


if __name__ == "__main__":
    # Teste NMS
    print("="*60)
    print("TESTE DE NMS")
    print("="*60)
    
    # Criar boxes de teste
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [105, 105, 205, 205],
        [300, 300, 400, 400]
    ], dtype=np.float32)
    
    scores = np.array([0.9, 0.8, 0.85, 0.95])
    
    print(f"\nBoxes originais: {len(boxes)}")
    print(f"Scores: {scores}")
    
    # NMS tradicional
    keep = nms(boxes, scores, threshold=0.5)
    print(f"\nNMS tradicional:")
    print(f"  Boxes mantidas: {len(keep)}")
    print(f"  Índices: {keep}")
    print(f"  Scores mantidos: {scores[keep]}")
    
    # Soft-NMS
    keep_soft, scores_soft = soft_nms(boxes, scores, threshold=0.5)
    print(f"\nSoft-NMS:")
    print(f"  Boxes mantidas: {len(keep_soft)}")
    print(f"  Índices: {keep_soft}")
    print(f"  Scores atualizados: {scores_soft}")
    
    print("\n✓ NMS testado com sucesso!")