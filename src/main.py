from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import onnxruntime as ort
import cv2
import numpy as np
from typing import List, Dict, Any
from src.utils import BoxXYXYCFormat, BoxYOLOFormat, convert_boxes_to_xyxyc, get_logger, non_max_suppression, get_unique_image_path, IMAGE_DIR, plot_and_save
from src.config import config

app = FastAPI(title="Human Detection API", description="API for human detection using YOLO model")
session = ort.InferenceSession(config.model_path)
input_name = session.get_inputs()[0].name
IOU_THRESHOLD = config.iou_threshold
CONF_THRESHOLD = config.confidence_threshold
logger = get_logger() 


def save_uploaded_files(files: List[UploadFile]) -> List[Path]:
    """Save uploaded files to disk and return their filenames."""
    uploaded_files_paths = []
    for file in files:
        try:
            file_path = get_unique_image_path(str(file.filename))
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files_paths.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {str(e)}")
    return uploaded_files_paths

def preprocess_images(filenames: List[Path]) -> List[np.ndarray]:
    """Load and preprocess images for model input."""
    images = []
    for filename in filenames:
        try:
            image = cv2.imread(str(filename))
            if image is None:
                raise ValueError(f"Could not read image: {filename}")
            image = cv2.resize(image, (640, 640))
            images.append(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {filename}: {str(e)}")
    return images

def prepare_model_input(images: List[np.ndarray]) -> np.ndarray:
    """Prepare the input tensor for the model."""
    try:
        input_tensor = np.array(images, dtype=np.float32).transpose(0, 3, 1, 2)
        return input_tensor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing model input: {str(e)}")

def process_detections(detections: np.ndarray) -> list[BoxXYXYCFormat]:
    """Process model detections and apply non-maximum suppression."""
    batch_size, _ , num_boxes = detections.shape
    
    yolo_boxes: list[BoxYOLOFormat] = []
    for i in range(batch_size):
        for j in range(num_boxes):
            x, y, width, height, confidence = detections[i, :, j]
            if confidence > CONF_THRESHOLD:
                box = BoxYOLOFormat(x=float(x), y=float(y), width=float(width), height=float(height), confidence=float(confidence))
                yolo_boxes.append(box)
    
    xyxyc_boxes: list[BoxXYXYCFormat] = convert_boxes_to_xyxyc(yolo_boxes)
    suppressed_boxes = non_max_suppression(xyxyc_boxes, IOU_THRESHOLD)
    
    return suppressed_boxes

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/check_health")
def check_health():
    return {"message": "OK"}

@app.get("/images")
def get_images():
    image_dir = Path(IMAGE_DIR)
    return [file.name for file in image_dir.glob("*.jpg")]

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Process uploaded images through the YOLO model and return detections.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Dict containing filenames, detections, and status message
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if files[0].content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Only JPEG images are supported")
        
        if len(files) > 1:
            raise HTTPException(status_code=400, detail="Only one file is allowed")

        # Save uploaded files
        uploaded_files = save_uploaded_files(files)
        
        # Preprocess images
        images = preprocess_images(uploaded_files)
        
        # Prepare model input
        input_tensor = prepare_model_input(images)
        
        # Run model inference
        result = session.run(None, {input_name: input_tensor})
        
        # Process detections
        boxes = process_detections(result[0])
        
        # Plot boxes on image
        plot_and_save(images[0],boxes, uploaded_files[0].name)

        return {
            "filenames": [file.name for file in uploaded_files],
            "objects_count": len(boxes),
            "boxes": boxes,
            "message": "Images processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images")
async def upload_images(files: list[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        file_path = get_unique_image_path(str(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)

    return {"filenames": uploaded_files, "message": "Images uploaded successfully"}

@app.get("/images/{image_name}")
def get_image(image_name: str):
    image_path = Path(IMAGE_DIR) / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")

