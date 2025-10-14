# MTCNN - Multi-task Cascaded Convolutional Networks

Implementação completa e detalhada do MTCNN em PyTorch para detecção facial e localização de landmarks.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Requisitos](#requisitos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Preparação dos Dados](#preparação-dos-dados)
- [Treinamento](#treinamento)
- [Inferência](#inferência)
- [Detalhes Técnicos](#detalhes-técnicos)
- [Resultados Esperados](#resultados-esperados)

---

## 🎯 Visão Geral

O MTCNN é uma arquitetura em cascata com três redes neurais convolucionais (P-Net, R-Net, O-Net) que trabalham em conjunto para:

1. **Detectar faces** em diferentes escalas
2. **Refinar detecções** progressivamente
3. **Localizar 5 landmarks faciais** (olhos, nariz, cantos da boca)

### Principais Características

✅ **Arquitetura em Cascata**: Três estágios progressivos (coarse-to-fine)  
✅ **Aprendizado Multi-tarefa**: Classificação + Regressão de BBox + Landmarks  
✅ **Online Hard Sample Mining**: Foco automático em amostras difíceis  
✅ **Tempo Real**: 99 fps em GPU, 16 fps em CPU  

---

## 💻 Requisitos

### Software
```bash
Python >= 3.8
PyTorch >= 2.0.0
CUDA >= 11.0 (opcional)
```

---

## 📁 Estrutura do Projeto

```
MTCNN/
├── config.py                    # Configurações globais
├── main.py                      # Script principal
├── requirements.txt             # Dependências
│
├── models/                      # Arquiteturas das redes
│   ├── pnet.py                 # Proposal Network
│   ├── rnet.py                 # Refinement Network
│   └── onet.py                 # Output Network
│
├── utils/                       # Utilitários
│   ├── bbox_utils.py           # NMS, IoU, calibração
│   ├── data_utils.py           # Dataset e parsers
│   └── image_utils.py          # Processamento de imagens
│
├── data_preprocessing/          # Geração de dados
│   ├── generate_pnet_data.py
│   ├── generate_rnet_data.py
│   └── generate_onet_data.py
│
├── training/                    # Scripts de treinamento
│   ├── train_pnet.py
│   ├── train_rnet.py
│   ├── train_onet.py
│   └── hard_sample_mining.py
│
├── inference/                   # Detecção
│   ├── detector.py             # Detector completo
│   └── visualization.py        # Visualização
│
├── data/                        # Dados
│   ├── raw/                    # Datasets originais
│   │   ├── WIDER_FACE/
│   │   └── CelebA/
│   └── processed/              # Dados processados
│       ├── pnet/
│       ├── rnet/
│       └── onet/
│
└── checkpoints/                 # Modelos treinados
    ├── pnet/
    ├── rnet/
    └── onet/
```

---

## 🚀 Instalação

### 1. Clone o Repositório
```bash
git clone <seu-repositorio>
cd mtcnn_project
```

### 2. Instale Dependências
```bash
pip install -r requirements.txt
```

### 3. Crie Estrutura de Diretórios
```bash
python config.py
```

---

## 📦 Preparação dos Dados

### 1. Download dos Datasets

#### WIDER FACE
```bash
# Download
wget http://shuoyang1213.me/WIDERFACE/WiderFace_BBXT.v2.tar.gz
wget http://shuoyang1213.me/WIDERFACE/WIDER_train.zip

# Extrair
tar -xzf WiderFace_BBXT.v2.tar.gz -C data/raw/WIDER_FACE/
unzip WIDER_train.zip -d data/raw/WIDER_FACE/
```

#### CelebA (Opcional, para landmarks)
```bash
# Download do Google Drive ou Kaggle
# Estrutura esperada:
data/raw/CelebA/
├── img_celeba/          # Imagens
└── list_landmarks_celeba.txt  # Anotações
```

### 2. Gerar Dados de Treinamento

#### P-Net Data
```bash
python main.py --mode prepare --stage pnet
```

**O que acontece:**
- Processa ~12,000 imagens do WIDER FACE
- Gera ~250,000 samples:
  - Negatives: ~100,000
  - Positives: ~70,000
  - Part faces: ~50,000
  - Landmarks: ~30,000
- Salva crops 12x12 em `data/processed/pnet/`
- Cria `pnet_train.txt` com anotações

**Tempo estimado:** 2-4 horas

---

## 🏋️ Treinamento

### Pipeline Sequencial

O treinamento segue uma **ordem específica** pois cada rede usa outputs das anteriores:

```
1. Treinar P-Net
2. Gerar dados R-Net usando P-Net
3. Treinar R-Net
4. Gerar dados O-Net usando P-Net + R-Net
5. Treinar O-Net
```

### 1. Treinar P-Net

```bash
python main.py --mode train --stage pnet
```

**Configurações (config.py):**
```python
PNET_INPUT_SIZE = 12
PNET_BATCH_SIZE = 512
PNET_EPOCHS = 30
PNET_LR = 0.001
PNET_CLS_WEIGHT = 1.0
PNET_BOX_WEIGHT = 0.5
PNET_LANDMARK_WEIGHT = 0.5
```

**Hard Sample Mining:**
- Ativado por padrão
- Ratio: 70% (top 70% das losses)
- Atualização online a cada batch

**Monitoramento:**
```bash
tensorboard --logdir runs/pnet
```

**Tempo estimado:** 4-6 horas (GPU), 20-24 horas (CPU)

### 2. Gerar Dados R-Net

```bash
python main.py --mode prepare --stage rnet
```

**O que acontece:**
- Carrega P-Net treinado
- Detecta faces em imagens do WIDER FACE
- Coleta detecções 24x24
- Gera `rnet_train.txt`

**Tempo estimado:** 3-5 horas

### 3. Treinar R-Net

```bash
python main.py --mode train --stage rnet
```

**Configurações:**
```python
RNET_INPUT_SIZE = 24
RNET_BATCH_SIZE = 256
RNET_EPOCHS = 25
RNET_LR = 0.001
```

**Tempo estimado:** 3-5 horas (GPU)

### 4. Gerar Dados O-Net

```bash
python main.py --mode prepare --stage onet
```

**Tempo estimado:** 4-6 horas

### 5. Treinar O-Net

```bash
python main.py --mode train --stage onet
```

**Configurações:**
```python
ONET_INPUT_SIZE = 48
ONET_BATCH_SIZE = 128
ONET_EPOCHS = 25
ONET_LR = 0.001
ONET_LANDMARK_WEIGHT = 1.0  # Peso MAIOR para landmarks
```

**Tempo estimado:** 4-6 horas (GPU)

### Pipeline Completo Automatizado

```bash
python main.py --mode full
```

**⚠️ Atenção:** Este comando executa TODO o pipeline sequencialmente.  
**Tempo total estimado:** 24-48 horas (depende do hardware)

---

## 🔍 Inferência

### Detecção em Uma Imagem

```bash
python main.py --mode detect --image path/to/image.jpg --output result.jpg
```

### Uso Programático

```python
from inference.detector import MTCNNDetector
import cv2

# Inicializar detector
detector = MTCNNDetector(
    pnet_path='checkpoints/pnet/pnet_final.pth',
    rnet_path='checkpoints/rnet/rnet_final.pth',
    onet_path='checkpoints/onet/onet_final.pth'
)

# Carregar imagem
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar
bboxes, scores, landmarks = detector.detect_faces(image_rgb, min_face_size=20)

print(f"Detectadas {len(bboxes)} faces")
# bboxes: [N, 4] - (x1, y1, x2, y2)
# scores: [N] - confiança
# landmarks: [N, 10] - 5 pontos (x,y)
```

### Processamento em Lote

```python
import glob

image_paths = glob.glob('images/*.jpg')

for img_path in image_paths:
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes, scores, landmarks = detector.detect_faces(image_rgb)
    
    # Processar resultados...
```

---

## 🔬 Detalhes Técnicos

### Arquiteturas das Redes

#### P-Net (Proposal Network)
```
Input: 12x12x3
Conv1: 10 filters, 3x3, stride=1
PReLU + MaxPool 2x2
Conv2: 16 filters, 3x3, stride=1
PReLU
Conv3: 32 filters, 3x3, stride=1
PReLU
Outputs:
  - Classification: 2 classes
  - BBox Regression: 4 offsets
  - Landmarks: 10 coordinates
```

**Parâmetros:** ~31,000

#### R-Net (Refinement Network)
```
Input: 24x24x3
Conv1: 28 filters, 3x3, stride=1
PReLU + MaxPool 3x3
Conv2: 48 filters, 3x3, stride=1
PReLU + MaxPool 3x3
Conv3: 64 filters, 2x2, stride=1
PReLU
FC1: 128 units
PReLU
Outputs (FC):
  - Classification: 2 classes
  - BBox Regression: 4 offsets
  - Landmarks: 10 coordinates
```

**Parâmetros:** ~98,000

#### O-Net (Output Network)
```
Input: 48x48x3
Conv1: 32 filters, 3x3, stride=1
PReLU + MaxPool 3x3
Conv2: 64 filters, 3x3, stride=1
PReLU + MaxPool 3x3
Conv3: 64 filters, 3x3, stride=1
PReLU + MaxPool 2x2
Conv4: 128 filters, 2x2, stride=1
PReLU
FC1: 256 units
PReLU
Outputs (FC):
  - Classification: 2 classes
  - BBox Regression: 4 offsets
  - Landmarks: 10 coordinates
```

**Parâmetros:** ~386,000

### Loss Function Multi-tarefa

```
L = α₁ * L_cls + α₂ * L_box + α₃ * L_landmark

onde:
- L_cls: Cross-Entropy Loss (classificação face/não-face)
- L_box: MSE Loss (regressão de bounding box)
- L_landmark: MSE Loss (localização de landmarks)
- α₁, α₂, α₃: pesos das tarefas

P-Net/R-Net: α₁=1.0, α₂=0.5, α₃=0.5
O-Net: α₁=1.0, α₂=0.5, α₃=1.0 (MAIOR peso para landmarks)
```

### Online Hard Sample Mining

**Algoritmo:**
```
1. Forward pass em TODO o batch
2. Calcular loss individual para cada sample
3. Ordenar losses em ordem decrescente
4. Selecionar top 70% como "hard samples"
5. Backward pass APENAS nos hard samples
```

**Benefícios:**
- Convergência 2-3x mais rápida
- Melhor generalização
- Sem necessidade de seleção manual
- Adaptativo ao treinamento

### Tipos de Samples

| Tipo | IoU | Uso | Loss Aplicada |
|------|-----|-----|---------------|
| Negative | < 0.3 | Classificação | Classification |
| Positive | > 0.65 | Cls + BBox | Classification + BBox |
| Part | 0.4-0.65 | Cls + BBox | Classification + BBox |
| Landmark | N/A | Landmarks | Landmark |

---

## 📊 Resultados Esperados

### Benchmarks

#### FDDB (Face Detection Data Set and Benchmark)
- **True Positive Rate @ 1000 FP:** ~95%
- **Comparação:** Superior a Viola-Jones, DPM, ACF

#### WIDER FACE
- **Easy:** ~94%
- **Medium:** ~93%
- **Hard:** ~85%

#### AFLW (Annotated Facial Landmarks in the Wild)
- **Mean Error (normalized):** ~1.5%

### Performance Computacional

| Hardware | FPS | Latência |
|----------|-----|----------|
| Nvidia Titan Black | 99 | 10ms |
| CPU (2.60GHz) | 16 | 62ms |
| Nvidia RTX 3090 | ~150 | 6-7ms |

### Curvas de Treinamento Típicas

**P-Net (30 epochs):**
- Loss inicial: ~2.5
- Loss final: ~0.3-0.5
- Melhor epoch: ~25

**R-Net (25 epochs):**
- Loss inicial: ~1.8
- Loss final: ~0.25-0.4

**O-Net (25 epochs):**
- Loss inicial: ~1.5
- Loss final: ~0.2-0.35

---

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"
**Solução:**
- Reduzir batch size em `config.py`
- P-Net: 512 → 256
- R-Net: 256 → 128
- O-Net: 128 → 64

### Erro: "Arquivo de anotações não encontrado"
**Solução:**
- Verificar paths em `config.py`
- Executar `python main.py --mode prepare --stage <stage>`

### Convergência lenta
**Soluções:**
- Verificar se hard sample mining está ativado
- Aumentar learning rate: 0.001 → 0.002
- Usar learning rate schedule (já implementado)

### Baixa qualidade de detecção
**Causas comuns:**
- Dados insuficientes ou desbalanceados
- Treinamento não convergiu
- Thresholds inadequados

**Soluções:**
- Gerar mais dados
- Treinar por mais épocas
- Ajustar thresholds em `config.py`

---

## 📚 Referências

**Paper Original:**
```
Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016).
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.
IEEE Signal Processing Letters, 23(10), 1499-1503.
```

**Datasets:**
- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- FDDB: http://vis-www.cs.umass.edu/fddb/
- AFLW: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/

---

## 📝 Licença

Este projeto é fornecido para fins educacionais e de pesquisa.

---

**Desenvolvido para pesquisa acadêmica em Reconhecimento Facial** 🎓