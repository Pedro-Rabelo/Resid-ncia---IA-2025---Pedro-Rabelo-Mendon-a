__version__ = '1.0.0'

try:
    from .detector import MTCNNDetector
except ImportError:
    MTCNNDetector = None

try:
    from .visualization import (
        draw_detections,
        draw_detections_matplotlib,
        visualize_face_crops,
        create_detection_grid,
        visualize_training_progress,
        compare_detections
    )
except ImportError:
    draw_detections = None
    draw_detections_matplotlib = None
    visualize_face_crops = None
    create_detection_grid = None
    visualize_training_progress = None
    compare_detections = None

__all__ = [
    'MTCNNDetector',
    'draw_detections',
    'draw_detections_matplotlib',
    'visualize_face_crops',
    'create_detection_grid',
    'visualize_training_progress',
    'compare_detections',
]