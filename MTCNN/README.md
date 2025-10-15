# MTCNN - Multi-task Cascaded Convolutional Networks

Implementação completa do MTCNN em PyTorch para detecção facial e localização de landmarks faciais.

---

## Sobre o Projeto

O MTCNN (Multi-task Cascaded Convolutional Networks) é uma arquitetura de deep learning em cascata composta por três redes neurais convolucionais (P-Net, R-Net, O-Net) que trabalham sequencialmente para realizar detecção facial robusta e localização precisa de landmarks faciais.

Este projeto é uma implementação completa do MTCNN do zero, desenvolvida durante o programa de residência, com foco em compreensão profunda dos fundamentos de detecção facial e aprendizado multi-tarefa.

**Implementação baseada em:** https://github.com/ipazc/mtcnn.git

### O que o MTCNN faz?

1. **Detecção de faces** em múltiplas escalas
2. **Refinamento progressivo** das detecções através de três estágios
3. **Localização de 5 landmarks faciais**: olhos esquerdo e direito, nariz, cantos da boca

---

## Características

- **Arquitetura em Cascata**: Três redes neurais (P-Net, R-Net, O-Net) trabalhando em pipeline coarse-to-fine
- **Aprendizado Multi-tarefa**: Cada rede aprende simultaneamente três tarefas: classificação de face, regressão de bounding box e localização de landmarks
- **Online Hard Sample Mining**: Técnica avançada para focar automaticamente em amostras difíceis durante o treinamento
- **Framework PyTorch**: Implementação moderna e otimizada para GPUs
- **Pipeline Completo**: Inclui pré-processamento de dados, treinamento e inferência
- **Datasets Padrão**: Suporte para WIDER FACE e CelebA

---

## Requisitos

### Software

```
Python >= 3.8
PyTorch >= 2.0.0
torchvision
numpy
opencv-python
Pillow
scikit-image
tqdm
matplotlib
tensorboard

---

---

## Detalhes Técnicos

### Arquitetura das Redes

#### P-Net (Proposal Network)

```
Input: 12x12x3 RGB image

Conv1: 10 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 2x2, stride 2

Conv2: 16 filters, kernel 3x3, stride 1
└── PReLU activation

Conv3: 32 filters, kernel 3x3, stride 1
└── PReLU activation

Output Heads (Conv 1x1):
├── Classification: 2 classes (face/non-face)
├── BBox Regression: 4 offsets (dx, dy, dw, dh)
└── Landmarks: 10 coordinates (5 points × 2)
```

**Parâmetros:** ~31,000

**Função:** Gerar proposals iniciais rapidamente em múltiplas escalas

#### R-Net (Refinement Network)

```
Input: 24x24x3 RGB image

Conv1: 28 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 3x3, stride 2

Conv2: 48 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 3x3, stride 2

Conv3: 64 filters, kernel 2x2, stride 1
└── PReLU activation

FC1: 128 units
└── PReLU activation

Output Heads (FC):
├── Classification: 2 classes
├── BBox Regression: 4 offsets
└── Landmarks: 10 coordinates
```

**Parâmetros:** ~98,000

**Função:** Refinar proposals do P-Net e rejeitar falsos positivos

#### O-Net (Output Network)

```
Input: 48x48x3 RGB image

Conv1: 32 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 3x3, stride 2

Conv2: 64 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 3x3, stride 2

Conv3: 64 filters, kernel 3x3, stride 1
├── PReLU activation
└── MaxPool 2x2, stride 2

Conv4: 128 filters, kernel 2x2, stride 1
└── PReLU activation

FC1: 256 units
└── PReLU activation

Output Heads (FC):
├── Classification: 2 classes
├── BBox Regression: 4 offsets
└── Landmarks: 10 coordinates (refinamento final)
```

**Parâmetros:** ~386,000

**Função:** Refinamento final de detecções e localização precisa de landmarks

### Função de Loss Multi-tarefa

Cada rede é treinada com uma loss combinada:

```
L_total = α₁ × L_cls + α₂ × L_box + α₃ × L_landmark

onde:
- L_cls: Binary Cross-Entropy (classificação face/não-face)
- L_box: Smooth L1 Loss (regressão de bounding box)
- L_landmark: Smooth L1 Loss (localização de landmarks)
- α₁, α₂, α₃: pesos de balanceamento das tarefas
```

**Pesos utilizados:**

| Rede  | α₁ (cls) | α₂ (box) | α₃ (landmark) |
|-------|----------|----------|---------------|
| P-Net | 1.0      | 0.5      | 0.5           |
| R-Net | 1.0      | 0.5      | 0.5           |
| O-Net | 1.0      | 0.5      | 1.0           |

**Nota:** O-Net tem peso maior para landmarks pois realiza o refinamento final.

### Online Hard Sample Mining

Técnica crucial para melhorar a performance das redes focando em amostras difíceis.

**Algoritmo:**

```
Para cada mini-batch:
1. Calcular loss individual para cada amostra
2. Ordenar amostras por loss (descendente)
3. Selecionar top 70% das amostras com maior loss
4. Calcular gradientes APENAS nas amostras selecionadas
5. Atualizar pesos da rede
```

**Benefícios:**
- Convergência mais rápida
- Melhor generalização
- Reduz overfitting em amostras fáceis

### Pipeline de Detecção

```
Imagem de Entrada
    │
    ├─→ [Pirâmide de Escalas]
    │       │
    │       └─→ P-Net (12x12) em cada escala
    │               │
    │               ├─→ Gera proposals
    │               └─→ NMS (threshold: 0.5)
    │
    ├─→ Resize proposals para 24x24
    │       │
    │       └─→ R-Net (24x24)
    │               │
    │               ├─→ Refina detecções
    │               ├─→ Rejeita falsos positivos
    │               └─→ NMS (threshold: 0.7)
    │
    └─→ Resize detecções para 48x48
            │
            └─→ O-Net (48x48)
                    │
                    ├─→ Refinamento final
                    ├─→ Localiza landmarks
                    └─→ NMS (threshold: 0.7)
                            │
                            └─→ [Saída Final]
```

---

### Casos de Uso

1. **Sistemas de Vigilância:** Detecção em tempo real de múltiplas faces
2. **Controle de Acesso:** Localização precisa para reconhecimento facial
3. **Análise de Emoções:** Landmarks para alinhamento facial
4. **Aplicações Mobile:** Versão leve para dispositivos móveis
5. **Pré-processamento:** Pipeline para modelos de reconhecimento facial

---

## Créditos

Esta implementação foi desenvolvida como parte do programa de residência técnica, com o objetivo de compreensão profunda dos fundamentos de detecção facial usando deep learning.

**Implementação baseada em:**
- Repositório original: https://github.com/ipazc/mtcnn.git
- Paper: "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" (Zhang et al., 2016)

---

## Licença

Este projeto é disponibilizado para fins educacionais e de pesquisa. Por favor, consulte as licenças dos datasets originais (WIDER FACE e CelebA) antes de usar em aplicações comerciais.

MIT License - Ver arquivo LICENSE para detalhes.

---

**Nota:** Este README documenta uma implementação educacional do MTCNN. Para uso em produção, considere otimizações adicionais e testes extensivos nos seus dados específicos.