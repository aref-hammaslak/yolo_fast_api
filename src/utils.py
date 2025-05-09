import logging
from src.config import config
import cv2
import numpy as np
from pydantic import BaseModel
from datetime import datetime
import os
from pathlib import Path



IMAGE_DIR = config.image_dir
PLOTS_DIR = config.plots_dir

class BoxYOLOFormat(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    
class BoxXYXYCFormat(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    
def get_logger() -> logging.Logger:
    logger = logging.getLogger("human_detection")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

    
def plot_and_save(image: np.ndarray, boxes: list[BoxXYXYCFormat], image_name: str) -> None:
    """Plot boxes on image"""
    for box in boxes:
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 0, 255), 2)
    
    plot_dir = Path(PLOTS_DIR)
    plot_dir.mkdir(exist_ok=True)
    image_path = plot_dir / f"{image_name}.jpg"
    cv2.imwrite(str(image_path), image)
    
def convert_boxes_to_xyxyc(boxes: list[BoxYOLOFormat]) -> list[BoxXYXYCFormat]  :
    """Convert (x, y, width, height) to (x1, y1, x2, y2)"""
    xyxyc_boxes = []
    for box in boxes:
        x1 = box.x - box.width / 2
        y1 = box.y - box.height / 2
        x2 = box.x + box.width / 2
        y2 = box.y + box.height / 2
        xyxyc_boxes.append(BoxXYXYCFormat(x1=x1, y1=y1, x2=x2, y2=y2, confidence=box.confidence))  
    
    
    return xyxyc_boxes[0] if len(xyxyc_boxes) == 1 else xyxyc_boxes
    
def create_unique_image_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = Path(IMAGE_DIR) / f"images_{timestamp}"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir

def get_unique_image_path(image_name: str) -> Path:
    image_dir = create_unique_image_dir()
    for i in range(1000):
        image_path = image_dir / f"{image_name}_{i}.jpg"
        if not image_path.exists():
            return image_path
    raise Exception("Failed to create unique image path")

def iou(box1: BoxXYXYCFormat, box2: BoxXYXYCFormat) -> float:
    """Calculate IoU (Intersection over Union) between two bounding boxes."""

    # Compute intersection coordinates
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)

    # Compute intersection width and height
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Compute areas of each box
    box1_area = max(0.0, box1.x2 - box1.x1) * max(0.0, box1.y2 - box1.y1)
    box2_area = max(0.0, box2.x2 - box2.x1) * max(0.0, box2.y2 - box2.y1)

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def non_max_suppression(boxes: list[BoxXYXYCFormat], iou_threshold: float = 0.5) -> list[BoxXYXYCFormat]:
    """Apply NMS to remove overlapping boxes.

    Args:
        boxes: list of Box objects
        iou_threshold: float, default 0.5

    Returns:
        list of Box objects
    """
    if not boxes:
        return []

    # Sort boxes by confidence score in descending order
    boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
    keep = []

    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        filtered_boxes = []
        for box in boxes:
            if iou(current, box) <= iou_threshold:
                filtered_boxes.append(box)
        boxes = filtered_boxes

    return keep