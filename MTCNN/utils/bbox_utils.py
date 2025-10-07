import numpy as np
import torch

def nms(boxes, scores, threshold=0.5, mode='union'):
    """
    Non-Maximum Suppression (NMS)
    
    Args:
        boxes: numpy array [N, 4] formato (x1, y1, x2, y2)
        scores: numpy array [N] scores de confiança
        threshold: float, IoU threshold para supressão
        mode: 'union' ou 'min' para cálculo de IoU
    
    Returns:
        keep: indices das boxes mantidas
    """
    if boxes.shape[0] == 0:
        return np.array([])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Ordenar por score decrescente
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calcular IoU com boxes restantes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if mode == 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:  # union
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Manter apenas boxes com IoU < threshold
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def convert_to_square(bboxes):
    """
    Converte bounding boxes retangulares para quadradas
    Útil para redimensionamento uniforme
    
    Args:
        bboxes: numpy array [N, 4] ou [N, 5] formato (x1, y1, x2, y2) ou (..., score)
    
    Returns:
        square_bboxes: bboxes quadradas
    """
    square_bboxes = bboxes.copy()
    
    h = bboxes[:, 3] - bboxes[:, 1] + 1
    w = bboxes[:, 2] - bboxes[:, 0] + 1
    max_side = np.maximum(h, w)
    
    square_bboxes[:, 0] = bboxes[:, 0] + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = bboxes[:, 1] + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1
    
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """
    Calibra bounding boxes usando offsets preditos
    
    Args:
        bboxes: numpy array [N, 4] formato (x1, y1, x2, y2)
        offsets: numpy array [N, 4] formato (dx1, dy1, dx2, dy2)
    
    Returns:
        calibrated_boxes: boxes calibradas
    """
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    
    # Aplicar offsets
    x1 = x1 + offsets[:, 0] * w
    y1 = y1 + offsets[:, 1] * h
    x2 = x2 + offsets[:, 2] * w
    y2 = y2 + offsets[:, 3] * h
    
    calibrated_boxes = np.stack([x1, y1, x2, y2], axis=1)
    return calibrated_boxes


def compute_iou(box, boxes):
    """
    Calcula IoU entre uma box e várias outras
    
    Args:
        box: numpy array [4] formato (x1, y1, x2, y2)
        boxes: numpy array [N, 4]
    
    Returns:
        iou: numpy array [N] valores de IoU
    """
    # Coordenadas da box
    box_x1, box_y1, box_x2, box_y2 = box
    box_area = (box_x2 - box_x1 + 1) * (box_y2 - box_y1 + 1)
    
    # Coordenadas das boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Interseção
    xx1 = np.maximum(box_x1, x1)
    yy1 = np.maximum(box_y1, y1)
    xx2 = np.minimum(box_x2, x2)
    yy2 = np.minimum(box_y2, y2)
    
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    
    # IoU
    iou = inter / (box_area + areas - inter)
    return iou


def pad_bbox(bbox, img_shape):
    """
    Garante que bbox está dentro dos limites da imagem
    
    Args:
        bbox: numpy array [4] formato (x1, y1, x2, y2)
        img_shape: tuple (height, width)
    
    Returns:
        padded_bbox: bbox ajustada
        pad: padding aplicado [top, bottom, left, right]
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calcular padding necessário
    pad_top = max(0, -int(y1))
    pad_bottom = max(0, int(y2) - h + 1)
    pad_left = max(0, -int(x1))
    pad_right = max(0, int(x2) - w + 1)
    
    # Ajustar coordenadas
    x1_new = max(0, int(x1))
    y1_new = max(0, int(y1))
    x2_new = min(w - 1, int(x2))
    y2_new = min(h - 1, int(y2))
    
    padded_bbox = np.array([x1_new, y1_new, x2_new, y2_new])
    pad = [pad_top, pad_bottom, pad_left, pad_right]
    
    return padded_bbox, pad


def generate_bboxes(cls_map, reg_map, scale, threshold):
    """
    Gera bounding boxes a partir de outputs do P-Net
    
    Args:
        cls_map: numpy array [H, W] - mapa de probabilidades
        reg_map: numpy array [4, H, W] - offsets de regressão
        scale: float - fator de escala da imagem
        threshold: float - threshold de confiança
    
    Returns:
        bboxes: numpy array [N, 9] formato (x1, y1, x2, y2, score, dx1, dy1, dx2, dy2)
    """
    stride = 2
    cell_size = 12
    
    # Encontrar posições com score > threshold
    t_index = np.where(cls_map > threshold)
    
    if t_index[0].size == 0:
        return np.array([])
    
    # Offsets de regressão
    dx1, dy1, dx2, dy2 = reg_map[:, t_index[0], t_index[1]]
    
    # Scores
    score = cls_map[t_index[0], t_index[1]]
    
    # Bounding boxes antes de regressão
    x1 = np.round((stride * t_index[1] + 1) / scale)
    y1 = np.round((stride * t_index[0] + 1) / scale)
    x2 = np.round((stride * t_index[1] + 1 + cell_size) / scale)
    y2 = np.round((stride * t_index[0] + 1 + cell_size) / scale)
    
    # Concatenar
    bboxes = np.stack([x1, y1, x2, y2, score, dx1, dy1, dx2, dy2], axis=1)
    
    return bboxes


def landmarks_to_absolute(landmarks, bboxes):
    """
    Converte landmarks relativos (0-1) para coordenadas absolutas
    
    Args:
        landmarks: numpy array [N, 10] - landmarks relativos
        bboxes: numpy array [N, 4] - bounding boxes (x1, y1, x2, y2)
    
    Returns:
        absolute_landmarks: numpy array [N, 10] - landmarks absolutos
    """
    x1 = bboxes[:, 0:1]
    y1 = bboxes[:, 1:2]
    w = (bboxes[:, 2:3] - x1 + 1)
    h = (bboxes[:, 3:4] - y1 + 1)
    
    # Landmarks: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    absolute_landmarks = np.zeros_like(landmarks)
    
    # x coordinates (índices pares)
    absolute_landmarks[:, 0::2] = landmarks[:, 0::2] * w + x1
    
    # y coordinates (índices ímpares)
    absolute_landmarks[:, 1::2] = landmarks[:, 1::2] * h + y1
    
    return absolute_landmarks


if __name__ == "__main__":
    print("="*60)
    print("TESTE DE BBOX UTILS")
    print("="*60)
    
    # Test NMS
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [300, 300, 400, 400]
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.95])
    
    keep = nms(boxes, scores, threshold=0.5)
    print(f"\nNMS test:")
    print(f"Input boxes: {boxes.shape[0]}")
    print(f"Kept boxes: {len(keep)}")
    print(f"Kept indices: {keep}")
    
    # Test square conversion
    square_boxes = convert_to_square(boxes)
    print(f"\nSquare conversion:")
    print(f"Original: {boxes[0]}")
    print(f"Square: {square_boxes[0]}")
    
    # Test IoU
    box = boxes[0]
    ious = compute_iou(box, boxes[1:])
    print(f"\nIoU test:")
    print(f"Reference box: {box}")
    print(f"IoUs with other boxes: {ious}")
    
    # Test calibration
    offsets = np.array([[0.1, 0.1, -0.1, -0.1],
                        [0.05, 0.05, -0.05, -0.05],
                        [0, 0, 0, 0]])
    calibrated = calibrate_box(boxes, offsets)
    print(f"\nCalibration test:")
    print(f"Original box: {boxes[0]}")
    print(f"Offset: {offsets[0]}")
    print(f"Calibrated: {calibrated[0]}")
    
    print("\n✓ Bbox utils testados com sucesso!")