# MTCNN Quick Start Guide 🚀

Guia passo-a-passo para implementar o MTCNN do zero.

---

## ✅ CHECKLIST DE IMPLEMENTAÇÃO

### Fase 1: Setup Inicial (30 min)

- [ ] **1.1** Clonar/criar estrutura de pastas
```bash
mkdir mtcnn_project && cd mtcnn_project
```

- [ ] **1.2** Criar `requirements.txt` e instalar
```bash
pip install torch torchvision numpy opencv-python Pillow scikit-image tqdm matplotlib tensorboard
```

- [ ] **1.3** Criar `config.py` com todas as configurações
  - Paths dos datasets
  - Hyperparâmetros de treinamento
  - Thresholds de inferência

- [ ] **1.4** Criar estrutura de diretórios
```bash
python config.py  # Script que cria todos os diretórios
```

---

### Fase 2: Implementação das Redes (2-3 horas)

- [ ] **2.1** Implementar `models/pnet.py`
  - Arquitetura da rede
  - Loss function multi-tarefa
  - Método forward e predict

- [ ] **2.2** Implementar `models/rnet.py`
  - Similar ao P-Net mas com FC layers
  - Input 24x24

- [ ] **2.3** Implementar `models/onet.py`
  - Rede mais profunda
  - Input 48x48
  - Peso maior para landmarks

- [ ] **2.4** Testar cada modelo individualmente
```python
python models/pnet.py  # Deve rodar sem erros
python models/rnet.py
python models/onet.py
```

**✓ Checkpoint:** Todas as redes devem fazer forward pass sem erros

---

### Fase 3: Utilitários (2 horas)

- [ ] **3.1** Implementar `utils/bbox_utils.py`
  - NMS (Non-Maximum Suppression)
  - IoU computation
  - Box calibration
  - Convert to square

- [ ] **3.2** Implementar `utils/data_utils.py`
  - MTCNNDataset class
  - Parse WIDER FACE annotations
  - Parse CelebA landmarks

- [ ] **3.3** Testar utilitários
```python
python utils/bbox_utils.py
python utils/data_utils.py
```

**✓ Checkpoint:** NMS funciona corretamente, IoU calculado

---

### Fase 4: Download de Datasets (1-2 horas)

- [ ] **4.1** Download WIDER FACE
```bash
# Training images (~1.5GB)
wget http://shuoyang1213.me/WIDERFACE/WIDER_train.zip
unzip WIDER_train.zip -d data/raw/WIDER_FACE/

# Annotations (~100MB)
wget http://shuoyang1213.me/WIDERFACE/WiderFace_BBXT.v2.tar.gz
tar -xzf WiderFace_BBXT.v2.tar.gz -C data/raw/WIDER_FACE/
```

- [ ] **4.2** Download CelebA (opcional)
```bash
# Via Kaggle ou Google Drive
# ~1.3GB de imagens
# Extrair em data/raw/CelebA/
```

- [ ] **4.3** Verificar estrutura
```
data/raw/
├── WIDER_FACE/
│   ├── WIDER_train/
│   │   └── images/
│   └── wider_face_split/
│       └── wider_face_train_bbx_gt.txt
└── CelebA/
    ├── img_celeba/
    └── list_landmarks_celeba.txt
```

**✓ Checkpoint:** Anotações podem ser lidas corretamente

---

### Fase 5: Geração de Dados P-Net (3-4 horas)

- [ ] **5.1** Implementar `data_preprocessing/generate_pnet_data.py`
  - Processar WIDER FACE
  - Gerar negatives (IoU < 0.3)
  - Gerar positives (IoU > 0.65)
  - Gerar part faces (0.4 < IoU < 0.65)
  - Gerar landmarks do CelebA

- [ ] **5.2** Executar geração
```bash
python main.py --mode prepare --stage pnet
```

- [ ] **5.3** Verificar output
```bash
ls data/processed/pnet/
# Deve conter:
# - positive/
# - negative/
# - part/
# - landmark/
# - pnet_train.txt
```

**✓ Checkpoint:** ~250k samples gerados, arquivo de anotação criado

---

### Fase 6: Treinamento P-Net (4-6 horas GPU)

- [ ] **6.1** Implementar `training/hard_sample_mining.py`
  - OnlineHardSampleMining class
  - Filter hard samples
  - Compute loss with mining

- [ ] **6.2** Implementar `training/train_pnet.py`
  - Training loop
  - Integração com hard mining
  - TensorBoard logging
  - Checkpoint saving

- [ ] **6.3** Executar treinamento
```bash
python main.py --mode train --stage pnet
```

- [ ] **6.4** Monitorar treinamento
```bash
tensorboard --logdir runs/pnet
```

**✓ Checkpoint:** Loss converge para ~0.3-0.5, modelo salvo

---

### Fase 7: R-Net Pipeline (6-8 horas)

- [ ] **7.1** Implementar `data_preprocessing/generate_rnet_data.py`
  - Usar P-Net treinado para detectar faces
  - Coletar detecções 24x24
  - Calcular IoU com ground truth
  - Salvar samples classificados

- [ ] **7.2** Gerar dados R-Net
```bash
python main.py --mode prepare --stage rnet
```

- [ ] **7.3** Implementar `training/train_rnet.py`
  - Similar ao P-Net mas input 24x24

- [ ] **7.4** Treinar R-Net
```bash
python main.py --mode train --stage rnet
```

**✓ Checkpoint:** R-Net treinado, loss ~0.25-0.4

---

### Fase 8: O-Net Pipeline (8-10 horas)

- [ ] **8.1** Implementar `data_preprocessing/generate_onet_data.py`
  - Usar P-Net + R-Net em cascata
  - Coletar detecções finais 48x48

- [ ] **8.2** Gerar dados O-Net
```bash
python main.py --mode prepare --stage onet
```

- [ ] **8.3** Implementar `training/train_onet.py`
  - Peso MAIOR para landmarks (1.0)

- [ ] **8.4** Treinar O-Net
```bash
python main.py --mode train --stage onet
```

**✓ Checkpoint:** O-Net treinado, loss ~0.2-0.35

---

### Fase 9: Implementação do Detector (2-3 horas)

- [ ] **9.1** Implementar `inference/detector.py`
  - MTCNNDetector class
  - Stage 1: P-Net (image pyramid)
  - Stage 2: R-Net (refinement)
  - Stage 3: O-Net (final + landmarks)
  - NMS entre estágios

- [ ] **9.2** Implementar `inference/visualization.py`
  - Desenhar bounding boxes
  - Desenhar landmarks
  - Salvar resultados

- [ ] **9.3** Testar detector
```bash
python main.py --mode detect --image test.jpg --output result.jpg
```

**✓ Checkpoint:** Faces detectadas com landmarks

---

### Fase 10: Validação e Testes (2-3 horas)

- [ ] **10.1** Testar em imagens variadas
  - Single face
  - Multiple faces
  - Diferentes iluminações
  - Diferentes poses

- [ ] **10.2** Benchmark de velocidade
```python
import time
start = time.time()
bboxes, scores, landmarks = detector.detect_faces(image)
print(f"Tempo: {time.time() - start:.3f}s")
```

- [ ] **10.3** Ajustar thresholds se necessário
```python
# Em config.py
PNET_THRESHOLD = 0.6  # Reduzir para mais detecções
RNET_THRESHOLD = 0.7
ONET_THRESHOLD = 0.7
```

**✓ Checkpoint:** Detector funcional e rápido

---

## 📊 TIMELINE ESTIMADO

| Fase | Descrição | Tempo CPU | Tempo GPU |
|------|-----------|-----------|-----------|
| 1 | Setup | 30 min | 30 min |
| 2 | Redes | 3 horas | 3 horas |
| 3 | Utils | 2 horas | 2 horas |
| 4 | Download | 2 horas | 2 horas |
| 5 | Dados P-Net | 4 horas | 4 horas |
| 6 | Train P-Net | 20 horas | 5 horas |
| 7 | R-Net Pipeline | 25 horas | 7 horas |
| 8 | O-Net Pipeline | 28 horas | 9 horas |
| 9 | Detector | 3 horas | 3 horas |
| 10 | Testes | 3 horas | 3 horas |
| **TOTAL** | | **~90 horas** | **~38 horas** |

---

## 🎯 MILESTONES CRÍTICOS

### Milestone 1: Redes Implementadas ✅
**Verificação:**
```python
from models.pnet import PNet
from models.rnet import RNet
from models.onet import ONet

p = PNet()
r = RNet()
o = ONet()

import torch
x_p = torch.randn(1, 3, 12, 12)
x_r = torch.randn(1, 3, 24, 24)
x_o = torch.randn(1, 3, 48, 48)

# Deve rodar sem erros
cls_p, box_p, lmk_p = p(x_p)
cls_r, box_r, lmk_r = r(x_r)
cls_o, box_o, lmk_o = o(x_o)
print("✓ Todas as redes funcionando!")
```

### Milestone 2: P-Net Treinado ✅
**Verificação:**
```bash
# Checkpoint existe
ls checkpoints/pnet/pnet_final.pth

# Loss convergiu
tensorboard --logdir runs/pnet
# Verificar que loss < 0.5
```

### Milestone 3: Pipeline Completo ✅
**Verificação:**
```bash
# Todos os checkpoints existem
ls checkpoints/*/

# Detector funciona
python main.py --mode detect --image test.jpg
```

---

## 🚨 PROBLEMAS COMUNS E SOLUÇÕES

### 1. "CUDA out of memory"
```python
# Reduzir batch sizes em config.py
PNET_BATCH_SIZE = 256  # era 512
RNET_BATCH_SIZE = 128  # era 256
ONET_BATCH_SIZE = 64   # era 128
```

### 2. Loss não converge
```python
# Verificar learning rate
# Aumentar se muito lento, diminuir se oscilando
PNET_LR = 0.0005  # testar valores

# Verificar hard sample mining
ONLINE_MINING = True  # deve estar True
HARD_SAMPLE_RATIO = 0.7  # testar 0.6-0.8
```

### 3. Poucos dados gerados
```python
# Aumentar samples per image em config.py
PNET_SAMPLES_PER_IMAGE = {
    'positive': 80,   # era 50
    'negative': 80,   # era 50
    'part': 80,       # era 50
    'landmark': 20    # era 10
}
```

### 4. Detecções de baixa qualidade
```python
# Ajustar thresholds
PNET_THRESHOLD = 0.5   # reduzir para mais detecções
RNET_THRESHOLD = 0.6
ONET_THRESHOLD = 0.6

# Ajustar NMS threshold
PNET_NMS_THRESHOLD = 0.5  # reduzir para menos supressão
```

---

## 📈 MÉTRICAS DE SUCESSO

### Durante Treinamento
- **P-Net Loss:** deve atingir < 0.5
- **R-Net Loss:** deve atingir < 0.4
- **O-Net Loss:** deve atingir < 0.35
- **Hard Mining:** ratio ~70%, kept samples consistente

### Após Inferência
- **Faces detectadas:** > 90% das faces visíveis
- **False positives:** < 5% das detecções
- **Landmarks:** erro < 5 pixels em faces 100x100
- **Velocidade:** > 10 FPS em CPU, > 50 FPS em GPU

---

## 🎓 PRÓXIMOS PASSOS

Após implementação completa:

1. **Otimização**
   - [ ] Quantização INT8
   - [ ] ONNX export
   - [ ] TensorRT optimization

2. **Experimentos**
   - [ ] Testar diferentes backbones
   - [ ] Adicionar attention mechanisms
   - [ ] Data augmentation avançada

3. **Aplicações**
   - [ ] Face recognition pipeline
   - [ ] Real-time video processing
   - [ ] Mobile deployment

---

**Boa sorte na implementação! 🚀**

Para dúvidas ou problemas, revise o README.md completo.