# Reconhecimento Facial - Pré-processamento e Treinamento

## MTCNN

O MTCNN (Multi-task Cascaded Convolutional Networks) é uma arquitetura de rede neural em cascata composta por três estágios sequenciais para detecção e alinhamento facial. O primeiro estágio (P-Net) gera rapidamente candidatos a faces em diferentes escalas da imagem. O segundo estágio (R-Net) refina esses candidatos, eliminando falsos positivos. O terceiro estágio (O-Net) produz a detecção final da face junto com cinco landmarks faciais (olhos, nariz e cantos da boca). Esta abordagem multi-estágio permite detecção robusta de faces em diferentes poses, iluminações e oclusões, sendo amplamente utilizada como etapa de pré-processamento em sistemas de reconhecimento facial.

## Pipeline

O Pipeline implementa um sistema completo de reconhecimento facial baseado em MobileNetV3 com aprendizado multi-tarefa. O sistema realiza três funções principais: pré-processamento de datasets faciais com detecção e alinhamento automático usando RetinaFace, treinamento de modelos de reconhecimento facial com uma arquitetura que combina classificação de identidades e regressão de landmarks faciais como tarefa auxiliar, e inferência para extração de embeddings e comparação de faces. O modelo utiliza MobileNetV3-Large como backbone e é treinado no dataset VGGFace2, com validação no benchmark LFW. A abordagem multi-tarefa melhora a qualidade dos embeddings ao forçar o modelo a aprender representações que são úteis tanto para identificação quanto para localização precisa de características faciais.
