import cv2
import numpy as np
from PIL import Image
import random


def load_image(image_path, color_mode='RGB'):
    """
    Carrega imagem de forma robusta
    
    Args:
        image_path: path da imagem
        color_mode: 'RGB', 'BGR' ou 'GRAY'
    
    Returns:
        image: numpy array
    """
    try:
        if color_mode == 'RGB':
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color_mode == 'BGR':
            return cv2.imread(image_path)
        elif color_mode == 'GRAY':
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Erro ao carregar {image_path}: {e}")
        return None


def resize_image(image, size, interpolation=cv2.INTER_LINEAR):
    """
    Redimensiona imagem
    
    Args:
        image: numpy array
        size: (width, height) ou int (quadrado)
        interpolation: método de interpolação
    
    Returns:
        resized_image: numpy array
    """
    if isinstance(size, int):
        size = (size, size)
    
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_image(image, mean=127.5, std=128.0):
    """
    Normaliza imagem para [-1, 1]
    
    Args:
        image: numpy array [H, W, C] em [0, 255]
        mean: valor médio
        std: desvio padrão
    
    Returns:
        normalized: numpy array em [-1, 1]
    """
    return (image.astype(np.float32) - mean) / std


def denormalize_image(image, mean=127.5, std=128.0):
    """
    Reverte normalização
    
    Args:
        image: numpy array normalizado
    
    Returns:
        denormalized: numpy array em [0, 255]
    """
    image = image * std + mean
    return np.clip(image, 0, 255).astype(np.uint8)


def random_crop(image, crop_size):
    """
    Crop aleatório
    
    Args:
        image: numpy array [H, W, C]
        crop_size: (height, width)
    
    Returns:
        cropped: numpy array
        (x, y): posição do crop
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        return image, (0, 0)
    
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    
    cropped = image[y:y+crop_h, x:x+crop_w]
    
    return cropped, (x, y)


def center_crop(image, crop_size):
    """
    Crop central
    
    Args:
        image: numpy array [H, W, C]
        crop_size: (height, width)
    
    Returns:
        cropped: numpy array
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    
    return image[y:y+crop_h, x:x+crop_w]


def random_flip(image, prob=0.5, horizontal=True):
    """
    Flip aleatório
    
    Args:
        image: numpy array
        prob: probabilidade
        horizontal: se True, flip horizontal; se False, vertical
    
    Returns:
        flipped_image: numpy array
        flipped: bool indicando se aplicou flip
    """
    if random.random() < prob:
        if horizontal:
            return cv2.flip(image, 1), True
        else:
            return cv2.flip(image, 0), True
    return image, False


def adjust_brightness(image, delta_range=(-30, 30)):
    """
    Ajusta brilho
    
    Args:
        image: numpy array
        delta_range: range de ajuste
    
    Returns:
        adjusted: numpy array
    """
    delta = random.uniform(*delta_range)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def adjust_contrast(image, alpha_range=(0.8, 1.2)):
    """
    Ajusta contraste
    
    Args:
        image: numpy array
        alpha_range: range de ajuste
    
    Returns:
        adjusted: numpy array
    """
    alpha = random.uniform(*alpha_range)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def add_gaussian_noise(image, mean=0, std=10):
    """
    Adiciona ruído gaussiano
    
    Args:
        image: numpy array
        mean: média do ruído
        std: desvio padrão
    
    Returns:
        noisy: numpy array
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(image, kernel_size=(3, 3), sigma=0):
    """
    Aplica blur gaussiano
    
    Args:
        image: numpy array
        kernel_size: tamanho do kernel
        sigma: sigma do gaussiano
    
    Returns:
        blurred: numpy array
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def augment_image(image, config=None):
    """
    Aplica augmentação completa
    
    Args:
        image: numpy array [H, W, C]
        config: dict com configurações de augmentação
    
    Returns:
        augmented: numpy array
    """
    if config is None:
        config = {
            'flip_prob': 0.5,
            'brightness': True,
            'contrast': True,
            'blur_prob': 0.1,
            'noise_prob': 0.1
        }
    
    img = image.copy()
    
    # Horizontal flip
    if random.random() < config.get('flip_prob', 0.5):
        img = cv2.flip(img, 1)
    
    # Brightness
    if config.get('brightness', True) and random.random() < 0.5:
        img = adjust_brightness(img)
    
    # Contrast
    if config.get('contrast', True) and random.random() < 0.5:
        img = adjust_contrast(img)
    
    # Gaussian blur
    if random.random() < config.get('blur_prob', 0.1):
        img = gaussian_blur(img)
    
    # Gaussian noise
    if random.random() < config.get('noise_prob', 0.1):
        img = add_gaussian_noise(img)
    
    return img


def create_image_pyramid(image, min_size, scale_factor=0.709):
    """
    Cria pirâmide de imagens para detecção multi-escala
    
    Args:
        image: numpy array [H, W, C]
        min_size: tamanho mínimo da menor dimensão
        scale_factor: fator de escala entre níveis
    
    Returns:
        pyramid: lista de (image, scale)
    """
    h, w = image.shape[:2]
    min_length = min(h, w)
    
    scales = []
    current_scale = 1.0
    
    while min_length * current_scale >= min_size:
        scales.append(current_scale)
        current_scale *= scale_factor
    
    pyramid = []
    for scale in scales:
        new_h = int(h * scale)
        new_w = int(w * scale)
        scaled_img = cv2.resize(image, (new_w, new_h), 
                               interpolation=cv2.INTER_LINEAR)
        pyramid.append((scaled_img, scale))
    
    return pyramid


def pad_image(image, pad_size, pad_value=0):
    """
    Adiciona padding à imagem
    
    Args:
        image: numpy array
        pad_size: (top, bottom, left, right) ou int (igual em todos)
        pad_value: valor do padding
    
    Returns:
        padded: numpy array
    """
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size, pad_size, pad_size)
    
    top, bottom, left, right = pad_size
    
    return cv2.copyMakeBorder(image, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=pad_value)