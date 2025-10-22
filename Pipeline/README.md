# Pipeline de Reconhecimento Facial com Multi-Task Learning

## Visão Geral

Pipeline completo para treinamento e inferência de modelos de reconhecimento facial baseado em MobileNetV3 com aprendizado multi-tarefa (Multi-Task Learning). O sistema implementa uma arquitetura que combina classificação de identidades faciais com regressão de landmarks como tarefa auxiliar, melhorando a robustez e qualidade dos embeddings gerados.

### Características Principais

- Backbone MobileNetV3-Large otimizado para reconhecimento facial
- Aprendizado multi-tarefa com classificação e regressão de landmarks
- Detecção e alinhamento facial automatizado com RetinaFace
- Validação em benchmark LFW (Labeled Faces in the Wild)
- Suporte a treinamento em VGGFace2 dataset
- Inferência otimizada para CPU e GPU

## Arquitetura

### Modelo Multi-Task

O modelo utiliza MobileNetV3-Large como backbone com duas cabeças especializadas:

**Embedding Head**
- Extração de features para reconhecimento facial
- Dimensão de embedding: 512D
- Global Depthwise Convolution (GDC) layer

**Landmark Head**
- Predição de 5 landmarks faciais (olhos, nariz, cantos da boca)
- Saída: 10 valores normalizados (x, y para cada landmark)
- Arquitetura auxiliar para melhorar representações faciais

### Função de Perda

A loss total combina dois objetivos:

```
L_total = L_classification + λ × L_landmark
```

- **L_classification**: Margin Cosine Product (CosFace) para classificação de identidades
- **L_landmark**: SmoothL1 Loss ou Wing Loss para regressão de landmarks
- **λ**: Peso da tarefa auxiliar (padrão: 0.5)

## Instalação

### Requisitos

```bash
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.19.0
tqdm>=4.60.0
uniface  # Detector RetinaFace
```

### Instalação de Dependências

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
pip install uniface
```

## Uso

### 1. Pré-processamento do Dataset

O script de pré-processamento realiza detecção facial, validação, alinhamento e extração de landmarks.

```bash
python preprocess_with_landmarks.py \
    --input-root data/raw/vggface2_112x112 \
    --output-root data/train/vggface2_aligned_112x112 \
    --landmarks-json data/train/vggface2_landmarks.json \
    --min-size 20 \
    --min-images-per-class 2
```

**Parâmetros:**
- `--input-root`: Diretório com imagens originais do VGGFace2
- `--output-root`: Diretório de saída para imagens alinhadas (112x112)
- `--landmarks-json`: Arquivo JSON para armazenar landmarks normalizados
- `--min-size`: Tamanho mínimo da face em pixels (padrão: 20)
- `--min-images-per-class`: Mínimo de imagens por identidade (padrão: 2)

**Saída:**
- Imagens alinhadas em 112x112 pixels
- Arquivo JSON com landmarks normalizados
- Estatísticas de processamento

### 2. Treinamento

O script de treinamento implementa o loop completo com validação em LFW.

```bash
python train_multitask.py \
    --root data/train/vggface2_aligned_112x112 \
    --landmarks-json data/train/vggface2_landmarks.json \
    --batch-size 256 \
    --epochs 30 \
    --lr 0.1 \
    --landmark-weight 0.5 \
    --save-path weights/vggface2 \
    --lfw-root data/val
```

**Parâmetros de Dataset:**
- `--root`: Diretório com imagens alinhadas
- `--landmarks-json`: Arquivo JSON com landmarks
- `--train-split`: Proporção treino/validação (padrão: 0.8)
- `--min-images-per-class`: Mínimo de imagens por identidade (padrão: 2)

**Parâmetros de Modelo:**
- `--embedding-dim`: Dimensão do embedding (padrão: 512)
- `--landmark-weight`: Peso da loss de landmarks (padrão: 0.5)
- `--use-wing-loss`: Usar Wing Loss ao invés de SmoothL1

**Parâmetros de Treinamento:**
- `--batch-size`: Tamanho do batch (padrão: 256)
- `--epochs`: Número de épocas (padrão: 30)
- `--lr`: Learning rate inicial (padrão: 0.1)
- `--momentum`: Momentum do SGD (padrão: 0.9)
- `--weight-decay`: Weight decay (padrão: 5e-4)

**Learning Rate Scheduler:**
- `--milestones`: Épocas para reduzir LR (padrão: [10, 20, 25])
- `--gamma`: Fator de decaimento do LR (padrão: 0.1)

**Validação:**
- `--lfw-root`: Diretório do dataset LFW (padrão: data/val)
- `--eval-freq`: Frequência de avaliação em épocas (padrão: 1)

**Saída:**
- Checkpoint do último modelo: `mobilenetv3_vggface2_multitask_last.ckpt`
- Checkpoint do melhor modelo: `mobilenetv3_vggface2_multitask_best.ckpt`
- Logs de treinamento com loss, accuracy e LFW results

### 3. Inferência

O script de inferência suporta três modos de operação.

#### 3.1 Comparação de Duas Faces

Compara duas imagens e determina se são da mesma pessoa.

```bash
python inference_multitask.py \
    --mode compare \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_best.ckpt \
    --img1 path/to/image1.jpg \
    --img2 path/to/image2.jpg \
    --threshold 0.35
```

**Saída:**
- Valor de similaridade coseno
- Decisão binária (mesma pessoa ou não)
- Comparação com threshold

#### 3.2 Extração de Embedding

Extrai o embedding de uma única imagem.

```bash
python inference_multitask.py \
    --mode extract \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_best.ckpt \
    --img1 path/to/image.jpg
```

**Saída:**
- Vetor de embedding (512D)
- Norma L2 do embedding
- Status da detecção facial

#### 3.3 Processamento em Lote

Extrai embeddings de múltiplas imagens em uma pasta.

```bash
python inference_multitask.py \
    --mode batch \
    --checkpoint weights/vggface2/mobilenetv3_vggface2_multitask_best.ckpt \
    --folder path/to/images/ \
    --output embeddings.json
```

**Saída:**
- Arquivo JSON com embeddings de todas as imagens
- Relatório de sucesso/falha
- Estatísticas de processamento

**Parâmetros Gerais:**
- `--checkpoint`: Caminho para o checkpoint do modelo
- `--threshold`: Limiar de similaridade para comparação (padrão: 0.35)
- `--gpu`: ID da GPU ou -1 para CPU (padrão: 0)

## Pipeline de Pré-processamento

### Etapas de Processamento

1. **Detecção Facial**: RetinaFace detecta faces e landmarks
2. **Validação**: Verifica tamanho mínimo e qualidade dos landmarks
3. **Alinhamento**: Transformação afim para normalizar pose facial
4. **Normalização de Landmarks**: Coordenadas normalizadas para [0, 1]
5. **Salvamento**: Imagens alinhadas e landmarks em JSON

### Critérios de Validação

- Tamanho mínimo da face detectada
- Landmarks dentro da bounding box
- Ordem vertical correta (olhos > nariz > boca)
- Alinhamento horizontal dos olhos

### Template de Alinhamento

Posições padrão dos landmarks em face frontal normalizada (112x112):

```
Left Eye:     (38.29, 51.70)
Right Eye:    (73.53, 51.50)
Nose:         (56.03, 71.74)
Mouth Left:   (41.55, 92.37)
Mouth Right:  (70.73, 92.20)
```

## Formato do Checkpoint

Os checkpoints salvos contêm:

```python
{
    'epoch': int,                      # Época atual
    'model': OrderedDict,              # Estado do modelo
    'classification_head': OrderedDict,# Estado do MCP head
    'optimizer': dict,                 # Estado do optimizer
    'lr_scheduler': dict,              # Estado do scheduler
    'best_lfw_accuracy': float,        # Melhor accuracy no LFW
    'num_classes': int,                # Número de identidades
    'args': Namespace                  # Argumentos de treinamento
}
```

## Formato do Landmarks JSON

Estrutura do arquivo de landmarks:

```json
{
    "identity_name/image_name.jpg": [
        x1, y1,  // Left Eye
        x2, y2,  // Right Eye
        x3, y3,  // Nose
        x4, y4,  // Mouth Left
        x5, y5   // Mouth Right
    ]
}
```

Todos os valores são normalizados para o intervalo [0, 1].

## Dataset Esperado

### VGGFace2

Estrutura do dataset:

```
data/
├── raw/
│   └── vggface2_112x112/
│       ├── n000001/
│       │   ├── 0001_01.jpg
│       │   ├── 0002_01.jpg
│       │   └── ...
│       ├── n000002/
│       └── ...
├── train/
│   ├── vggface2_aligned_112x112/
│   └── vggface2_landmarks.json
```

- Aproximadamente 8.631 identidades
- Imagens pré-redimensionadas para 112x112 (opcional)
- Múltiplas imagens por identidade

### LFW (Validação)

```
├── val/
│    └── lfw/
│       ├── Aaron_Eckhart/
│       │   └── Aaron_Eckhart0001.jpg
│       ├── Aaron_Guiel/
│       └── ...
    └──pairs.txt
```

Dataset para validação de verificação facial:
- 6.000 pares de faces
- Split padrão para benchmark

## Métricas de Avaliação

### Treinamento

- **Loss Total**: Soma ponderada de classificação e landmarks
- **Loss de Classificação**: Cross-entropy via Margin Cosine Product
- **Loss de Landmarks**: SmoothL1 ou Wing Loss
- **Accuracy**: Precisão de classificação de identidades

### Validação

- **LFW Accuracy**: Taxa de verificação correta no dataset LFW
- **Curvas ROC**: True Positive Rate vs False Positive Rate

## Referências

### Arquitetura

- MobileNetV3: Searching for MobileNetV3 (Howard et al., 2019)
- CosFace: Large Margin Cosine Loss for Deep Face Recognition (Wang et al., 2018)

### Loss Functions

- Wing Loss for Robust Facial Landmark Localisation (Feng et al., 2018)
- Multi-task Learning Using Uncertainty (Kendall et al., 2018)

### Datasets

- VGGFace2: A dataset for recognising faces across pose and age (Cao et al., 2018)
- LFW: Labeled Faces in the Wild (Huang et al., 2007)

## Licença

Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

## Contato

Para dúvidas ou contribuições, entre em contato através do repositório do projeto.
