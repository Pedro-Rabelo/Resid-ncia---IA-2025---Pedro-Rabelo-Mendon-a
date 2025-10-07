__version__ = '1.0.0'

# BBox Utils
try:
    from .bbox_utils import (
        nms,
        compute_iou,
        calibrate_box,
        convert_to_square,
        pad_bbox,
        generate_bboxes,
        landmarks_to_absolute
    )
except ImportError:
    nms = None
    compute_iou = None
    calibrate_box = None
    convert_to_square = None
    pad_bbox = None
    generate_bboxes = None
    landmarks_to_absolute = None

# Data Utils
try:
    from .data_utils import (
        MTCNNDataset,
        parse_wider_face_annotation,
        parse_celeba_landmarks,
        random_crop_with_bbox,
        augment_image
    )
except ImportError:
    MTCNNDataset = None
    parse_wider_face_annotation = None
    parse_celeba_landmarks = None
    random_crop_with_bbox = None
    augment_image = None

# Image Utils (se existir)
try:
    from .image_utils import *
except ImportError:
    pass

__all__ = [
    # BBox Utils
    'nms',
    'compute_iou',
    'calibrate_box',
    'convert_to_square',
    'pad_bbox',
    'generate_bboxes',
    'landmarks_to_absolute',
    # Data Utils
    'MTCNNDataset',
    'parse_wider_face_annotation',
    'parse_celeba_landmarks',
    'random_crop_with_bbox',
    'augment_image',
]
