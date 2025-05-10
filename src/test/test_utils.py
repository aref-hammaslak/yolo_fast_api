import numpy as np
import pytest
from src.utils import convert_boxes_to_xyxyc, BoxXYXYCFormat, BoxYOLOFormat, iou, non_max_suppression, plot_and_save
from pathlib import Path

def test_convert_boxes_to_xyxyc():
    boxes = [BoxYOLOFormat(x=1, y=1, width=1, height=1, confidence=1)]
    conveted_box = convert_boxes_to_xyxyc(boxes)
    assert isinstance(conveted_box, BoxXYXYCFormat), "Converted box should be a BoxXYXYCFormat"
    assert conveted_box == BoxXYXYCFormat(x1=0.5, y1=0.5, x2=1.5, y2=1.5, confidence=1)

@pytest.mark.parametrize("boxes", [
    [BoxYOLOFormat(x=1, y=1, width=1, height=1, confidence=1), BoxYOLOFormat(x=2, y=2, width=2, height=2, confidence=2)]
])
def test_convert_boxes_to_xyxyc_list(boxes):
    conveted_box = convert_boxes_to_xyxyc(boxes)
    assert isinstance(conveted_box, list), "converted box should be a list"
    assert len(conveted_box) == 2, "converted box should have 2 elements"
    
    assert conveted_box == [
        BoxXYXYCFormat(x1=0.5, y1=0.5, x2=1.5, y2=1.5, confidence=1),
        BoxXYXYCFormat(x1=1, y1=1, x2=3, y2=3, confidence=2)]

@pytest.mark.parametrize("box1, box2, expected", [
    # No overlap
    (BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=1), BoxXYXYCFormat(x1=2, y1=2, x2=3, y2=3, confidence=2), 0.0),
    # Partial overlap (intersection area = 0.25, union area = 1.75, IoU = 0.25/1.75)
    (BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=1), BoxXYXYCFormat(x1=0.5, y1=0.5, x2=1.5, y2=1.5, confidence=2), 0.25/1.75),
    # One box inside another (intersection area = 1, union area = 1, IoU = 1.0)
    (BoxXYXYCFormat(x1=0, y1=0, x2=2, y2=2, confidence=1), BoxXYXYCFormat(x1=0.5, y1=0.5, x2=1.5, y2=1.5, confidence=2), 1.0/4.0),
    # Identical boxes (IoU = 1.0)
    (BoxXYXYCFormat(x1=1, y1=1, x2=3, y2=3, confidence=1), BoxXYXYCFormat(x1=1, y1=1, x2=3, y2=3, confidence=2), 1.0),
    # Edge touching (no overlap)
    (BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=1), BoxXYXYCFormat(x1=1, y1=1, x2=2, y2=2, confidence=2), 0.0),
])
def test_iou(box1, box2, expected):
    iou_value = iou(box1, box2)
    assert pytest.approx(iou_value, rel=1e-3) == expected

@pytest.mark.parametrize("boxes, expected", [
    # Two overlapping boxes, higher confidence kept
    (
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=2, y2=2, confidence=0.9),
            BoxXYXYCFormat(x1=1, y1=1, x2=3, y2=3, confidence=0.8)
        ],
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=2, y2=2, confidence=0.9),
            # The second box overlaps but IoU is less than 0.5, so both are kept
            BoxXYXYCFormat(x1=1, y1=1, x2=3, y2=3, confidence=0.8)
        ]
    ),
    # Two boxes, one inside the other, high IoU, only highest confidence kept
    (
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=4, y2=4, confidence=0.95),
            BoxXYXYCFormat(x1=0.1, y1=0.1, x2=3.9, y2=3.9, confidence=0.7)
        ],
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=4, y2=4, confidence=0.95)
        ]
    ),
    # Non-overlapping boxes, both kept
    (
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=0.6),
            BoxXYXYCFormat(x1=2, y1=2, x2=3, y2=3, confidence=0.5)
        ],
        [
            BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=0.6),
            BoxXYXYCFormat(x1=2, y1=2, x2=3, y2=3, confidence=0.5)
        ]
    ),
    # Identical boxes, only one kept
    (
        [
            BoxXYXYCFormat(x1=5, y1=5, x2=10, y2=10, confidence=0.8),
            BoxXYXYCFormat(x1=5, y1=5, x2=10, y2=10, confidence=0.7)
        ],
        [
            BoxXYXYCFormat(x1=5, y1=5, x2=10, y2=10, confidence=0.8)
        ]
    ),
])
def test_non_max_suppression_list(boxes, expected):
    nms_boxes = non_max_suppression(boxes, iou_threshold=0.5)
    assert len(nms_boxes) == len(expected)
    for nms_box, expected_box in zip(nms_boxes, expected):
        assert nms_box == expected_box

def test_plot_and_save():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [BoxXYXYCFormat(x1=0, y1=0, x2=1, y2=1, confidence=1)]
    plot_and_save(image, boxes, "test")
    test_image_path = Path("plots/test.jpg")
    assert test_image_path.exists()
    assert test_image_path.stat().st_size > 0
    test_image_path.unlink()
    
