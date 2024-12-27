from ultralytics import YOLO
import os
import sys
import torch

try:
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Model paths
    model_path = r"D:\DataSets\BoxDoorData\trainedModels\yolov8s_data_202412_small\best.pt"
    print(f"\nModel path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)
    print("Model loaded successfully")
    
    # Export model
    print("\nExporting model to ONNX...")
    model.export(format="onnx", opset=12)
    print("Export completed")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    raise