import torch
import numpy as np
import cv2
from models.pnet import PNet
from models.rnet import RNet
from models.onet import ONet
from utils.bbox_utils import nms, calibrate_box, convert_to_square, generate_bboxes, pad_bbox
from config import Config

class MTCNNDetector:
    """
    Detector MTCNN completo com três estágios em cascata
    """
    
    def __init__(self, pnet_path, rnet_path, onet_path, device=None):
        """
        Args:
            pnet_path: path para checkpoint do P-Net
            rnet_path: path para checkpoint do R-Net
            onet_path: path para checkpoint do O-Net
            device: torch device
        """
        self.device = device if device else Config.DEVICE
        
        # Carregar modelos
        print("Carregando modelos MTCNN...")
        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)
        
        self.pnet.load_state_dict(torch.load(pnet_path, map_location=self.device))
        self.rnet.load_state_dict(torch.load(rnet_path, map_location=self.device))
        self.onet.load_state_dict(torch.load(onet_path, map_location=self.device))
        
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        
        # Thresholds
        self.pnet_threshold = Config.PNET_THRESHOLD
        self.rnet_threshold = Config.RNET_THRESHOLD
        self.onet_threshold = Config.ONET_THRESHOLD
        
        print("✓ MTCNN carregado com sucesso!")
    
    def detect_faces(self, image, min_face_size=20):
        """
        Detecta faces em uma imagem
        
        Args:
            image: numpy array [H, W, 3] em RGB
            min_face_size: tamanho mínimo de face em pixels
        
        Returns:
            bboxes: numpy array [N, 4] - (x1, y1, x2, y2)
            scores: numpy array [N] - confidence scores
            landmarks: numpy array [N, 10] - 5 pontos x,y
        """
        h, w = image.shape[:2]
        
        # ========== STAGE 1: P-Net (Proposal) ==========
        print("\n[Stage 1] P-Net - Generating proposals...")
        bboxes_stage1 = self._stage1_pnet(image, min_face_size)
        
        if len(bboxes_stage1) == 0:
            return np.array([]), np.array([]), np.array([])
        
        print(f"  → {len(bboxes_stage1)} proposals geradas")
        
        # ========== STAGE 2: R-Net (Refinement) ==========
        print("\n[Stage 2] R-Net - Refining proposals...")
        bboxes_stage2 = self._stage2_rnet(image, bboxes_stage1)
        
        if len(bboxes_stage2) == 0:
            return np.array([]), np.array([]), np.array([])
        
        print(f"  → {len(bboxes_stage2)} faces refinadas")
        
        # ========== STAGE 3: O-Net (Output) ==========
        print("\n[Stage 3] O-Net - Final detection + landmarks...")
        bboxes, scores, landmarks = self._stage3_onet(image, bboxes_stage2)
        
        print(f"  → {len(bboxes)} faces finais detectadas")
        
        return bboxes, scores, landmarks
    
    def _stage1_pnet(self, image, min_face_size):
        """Stage 1: P-Net para gerar proposals"""
        h, w = image.shape[:2]
        
        # Calcular escalas para image pyramid
        min_length = min(h, w)
        min_detection_size = 12
        factor = 0.709  # factor de escala
        
        scales = []
        m = min_detection_size / min_face_size
        min_length *= m
        
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * (factor ** factor_count))
            min_length *= factor
            factor_count += 1
        
        # Processar cada escala
        all_boxes = []
        
        for scale in scales:
            # Resize image
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            img_resized = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_LINEAR)
            
            # Normalizar e converter para tensor
            img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                cls_prob, box_reg, _ = self.pnet.predict(img_tensor)
            
            # Converter para numpy
            cls_prob = cls_prob.cpu().numpy().squeeze()
            box_reg = box_reg.cpu().numpy().squeeze()
            
            # Gerar bounding boxes
            boxes = generate_bboxes(cls_prob, box_reg, scale, self.pnet_threshold)
            
            if len(boxes) > 0:
                all_boxes.append(boxes)
        
        if len(all_boxes) == 0:
            return np.array([])
        
        # Concatenar todas as boxes
        all_boxes = np.vstack(all_boxes)
        
        # NMS
        keep = nms(all_boxes[:, :5], all_boxes[:, 4], 0.7)
        all_boxes = all_boxes[keep]
        
        # Calibrar boxes com offsets
        bboxes = all_boxes[:, :4]
        offsets = all_boxes[:, 5:9]
        bboxes = calibrate_box(bboxes, offsets)
        
        # Converter para quadradas
        bboxes = convert_to_square(bboxes)
        
        return bboxes
    
    def _stage2_rnet(self, image, bboxes):
        """Stage 2: R-Net para refinar proposals"""
        num_boxes = len(bboxes)
        
        # Preparar patches
        patches = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(np.int32)
            
            # Pad se necessário
            bbox_padded, pad = pad_bbox(bbox, image.shape)
            x1, y1, x2, y2 = bbox_padded
            
            # Crop e resize
            img_crop = image[y1:y2+1, x1:x2+1]
            
            if pad != [0, 0, 0, 0]:
                img_crop = cv2.copyMakeBorder(img_crop, *pad, cv2.BORDER_CONSTANT)
            
            img_resized = cv2.resize(img_crop, (24, 24), interpolation=cv2.INTER_LINEAR)
            
            # Normalizar
            img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
            patches.append(img_tensor)
        
        # Batch processing
        patches = np.stack(patches)
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            cls_prob, box_reg, _ = self.rnet.predict(patches)
        
        cls_prob = cls_prob.cpu().numpy()
        box_reg = box_reg.cpu().numpy()
        
        # Filtrar por threshold
        keep_idx = cls_prob > self.rnet_threshold
        bboxes = bboxes[keep_idx]
        cls_prob = cls_prob[keep_idx]
        box_reg = box_reg[keep_idx]
        
        if len(bboxes) == 0:
            return np.array([])
        
        # Calibrar boxes
        bboxes = calibrate_box(bboxes, box_reg)
        
        # NMS
        bboxes_with_scores = np.column_stack([bboxes, cls_prob])
        keep = nms(bboxes_with_scores[:, :5], bboxes_with_scores[:, 4], 0.7)
        bboxes = bboxes[keep]
        
        # Converter para quadradas
        bboxes = convert_to_square(bboxes)
        
        return bboxes
    
    def _stage3_onet(self, image, bboxes):
        """Stage 3: O-Net para detecção final e landmarks"""
        num_boxes = len(bboxes)
        
        # Preparar patches
        patches = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(np.int32)
            
            # Pad se necessário
            bbox_padded, pad = pad_bbox(bbox, image.shape)
            x1, y1, x2, y2 = bbox_padded
            
            # Crop e resize
            img_crop = image[y1:y2+1, x1:x2+1]
            
            if pad != [0, 0, 0, 0]:
                img_crop = cv2.copyMakeBorder(img_crop, *pad, cv2.BORDER_CONSTANT)
            
            img_resized = cv2.resize(img_crop, (48, 48), interpolation=cv2.INTER_LINEAR)
            
            # Normalizar
            img_tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
            patches.append(img_tensor)
        
        # Batch processing
        patches = np.stack(patches)
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            cls_prob, box_reg, landmarks = self.onet.predict(patches)
        
        cls_prob = cls_prob.cpu().numpy()
        box_reg = box_reg.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
        
        # Filtrar por threshold
        keep_idx = cls_prob > self.onet_threshold
        bboxes = bboxes[keep_idx]
        cls_prob = cls_prob[keep_idx]
        box_reg = box_reg[keep_idx]
        landmarks = landmarks[keep_idx]
        
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Calibrar boxes
        bboxes = calibrate_box(bboxes, box_reg)
        
        # Converter landmarks para coordenadas absolutas
        w = bboxes[:, 2] - bboxes[:, 0] + 1
        h = bboxes[:, 3] - bboxes[:, 1] + 1
        
        landmarks_abs = np.zeros_like(landmarks)
        landmarks_abs[:, 0::2] = bboxes[:, 0:1] + landmarks[:, 0::2] * w[:, np.newaxis]
        landmarks_abs[:, 1::2] = bboxes[:, 1:2] + landmarks[:, 1::2] * h[:, np.newaxis]
        
        # NMS final
        bboxes_with_scores = np.column_stack([bboxes, cls_prob])
        keep = nms(bboxes_with_scores[:, :5], bboxes_with_scores[:, 4], 0.7)
        
        bboxes = bboxes[keep]
        scores = cls_prob[keep]
        landmarks = landmarks_abs[keep]
        
        return bboxes, scores, landmarks


if __name__ == "__main__":
    print("="*60)
    print("MTCNN DETECTOR")
    print("="*60)
    print("\nUso:")
    print("detector = MTCNNDetector(pnet_path, rnet_path, onet_path)")
    print("bboxes, scores, landmarks = detector.detect_faces(image)")