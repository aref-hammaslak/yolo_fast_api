import yaml
from pydantic import BaseModel



class Config(BaseModel):
    model_path: str
    iou_threshold: float
    confidence_threshold: float
    classes: list[str]
    image_dir: str
    plots_dir: str
    

with open("src/config/config.yaml", "r") as f:
    config = Config(**yaml.safe_load(f))

