from ultralytics import YOLO
import os
from pathlib import Path

def train_fall_detection_model():
    """
    Train YOLOv8 model for fall detection (Sleeping, Falling, Fallen).
    """
    # 1. Load a pretrained YOLO model (Using yolov8n-pose as fallback for demo)
    model = YOLO('yolov8n-pose.pt') 
    
    # 2. Define dataset config path
    data_yaml = Path("c:/Users/ADMIN/Documents/Fall_Detection_2/dataset/data.yaml")
    
    # 2. Training with SOTA 2025 optimizations (Production: 100 epochs)
    # Optimized for CPU Speed: accumulate=4, freeze=10, cos_lr=True
    results = model.train(
        data=str(data_yaml),
        epochs=100,         # Production level
        imgsz=640,
        batch=16,
        accumulate=4,       # Gradient Accumulation (simulates batch=64)
        freeze=10,          # Freeze backbone to speed up gradients
        cos_lr=True,        # Use Cosine Learning Rate (OneCycleLR variant)
        name='fall_detection_sota_fast',
        amp=True,          # Enable AMP (Mixed Precision)
        device='cpu',       # Hard-coded for demo, but ready for GPU
        optimizer='AdamW',  # Faster convergence than SGD
        patience=50,        # Early stopping if no improvement
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        degrees=15.0,
        translate=0.2,
        scale=0.6,
        perspective=0.0001,
        shear=2.0,
        blur=0.1,
        save=True,
        plots=True
    )
    
    # Final validation
    metrics = model.val()
    print(f"Final Validation Metrics: {metrics}")
    
    # Export for deployment
    model.export(format='onnx')
    print("✅ Training complete and model exported.")

if __name__ == "__main__":
    train_fall_detection_model()
