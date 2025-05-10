# YOLO Human Detection API

A FastAPI-based REST API for human detection using YOLO (You Only Look Once) object detection model. This API provides endpoints for uploading images, detecting humans, and retrieving detection results.

## Features

- Human detection using YOLO model
- RESTful API endpoints for image processing
- Support for JPEG image uploads
- Automatic image preprocessing and detection visualization
- Docker and Kubernetes support for easy deployment
- Configurable detection parameters (confidence threshold, IOU threshold)

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for orchestration)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aref-hammaslak/yolo_fast_api.git
cd yolo_fast_api
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your YOLO model file in the `weights` directory:

```bash
mkdir -p weights
# Copy your yolo11m.onnx model to weights/
```

## Configuration

The application can be configured through `src/config/config.yaml`:

```yaml
model_path: "weights/yolo11m.onnx"
iou_threshold: 0.5
confidence_threshold: 0.5
classes: ["person"]
image_dir: "images"
plots_dir: "plots"
```

## Usage

### Running Locally

Start the FastAPI server:

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Welcome message
- `GET /check_health`: Health check endpoint
- `GET /images`: List all processed images
- `POST /predict`: Upload and process an image for human detection
- `POST /images`: Upload images
- `GET /images/{image_name}`: Retrieve a specific image

### Example Usage

1. Upload and process an image:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@path/to/your/image.jpg"
```

2. View detection results:

```bash
curl -X GET "http://localhost:8000/images/processed_image.jpg"
```

## Docker Deployment

Build and run using Docker Compose:

```bash
docker-compose up --build
```

## Kubernetes Deployment

Deploy to Kubernetes:

```bash
kubectl apply -f kubernetes/
```

## Project Structure

```
yolo_fast_api/
├── src/
│   ├── main.py           # FastAPI application
│   ├── utils.py          # Utility functions
│   └── config/           # Configuration files
├── images/               # Uploaded images
├── plots/               # Detection visualization
├── weights/             # Model weights
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
