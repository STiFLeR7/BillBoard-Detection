import os
from ultralytics import YOLO

def main():
    # Paths
    ROOT_DIR = "D:/BillBoard-Detection"
    DATA_PATH = os.path.join(ROOT_DIR, "data", "data.yaml")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Select YOLOv8 variant
    model = YOLO("yolov8n.pt")  # nano version (fastest for edge)

    # Train
    results = model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=640,
        project=os.path.join(ROOT_DIR, "runs", "detect"),  # training logs
        name="train"  # folder name under runs/detect
    )

    # Save best weights to models/
    best_model_src = os.path.join(ROOT_DIR, "runs", "detect", "train", "weights", "best.pt")
    best_model_dst = os.path.join(MODELS_DIR, "best.pt")

    if os.path.exists(best_model_src):
        import shutil
        shutil.copy(best_model_src, best_model_dst)
        print(f"[INFO] Best model copied to {best_model_dst}")
    else:
        print("[WARNING] best.pt not found. Check training run.")

if __name__ == "__main__":
    main()
