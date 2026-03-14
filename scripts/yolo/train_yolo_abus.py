"""
Train YOLOv8 on ABUS lesion detection (2D slices).

Usage:
    # Basic training
    python train_yolo_abus.py --data abus_lesion.yaml --model yolov8m.pt --epochs 100

    # Resume training
    python train_yolo_abus.py --resume runs/detect/abus_lesion/weights/last.pt

    # Train with specific settings
    python train_yolo_abus.py --data abus_lesion.yaml --model yolov8l.pt \
        --epochs 200 --imgsz 640 --batch 16 --device 0

    # Multi-GPU training
    python train_yolo_abus.py --data abus_lesion.yaml --model yolov8m.pt \
        --device 0,1 --batch 32
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ABUS lesion detection")
    parser.add_argument("--data", type=str, default="abus_lesion.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained model (yolov8n/s/m/l/x.pt) or resume checkpoint")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (images will be letterboxed)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (-1 for auto-batch)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--name", type=str, default="abus_lesion",
                        help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--cache", action="store_true",
                        help="Cache images in RAM for faster training")
    parser.add_argument("--rect", action="store_true",
                        help="Use rectangular training (better for non-square images)")
    parser.add_argument("--cos_lr", action="store_true",
                        help="Use cosine learning rate scheduler")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project directory for saving results")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    # Load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Resolve data path
    data_path = args.data
    if not Path(data_path).is_absolute():
        # Look relative to this script's directory
        script_dir = Path(__file__).parent
        if (script_dir / data_path).exists():
            data_path = str(script_dir / data_path)

    # Training configuration
    # Tailored for medical imaging / ABUS characteristics:
    # - Lower mosaic probability (medical images have consistent structure)
    # - No color augmentation (grayscale ultrasound)
    # - Moderate geometric augmentation
    # - Higher confidence threshold for single-class detection
    train_args = dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        project=args.project,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,          # Final LR = lr0 * lrf
        cos_lr=args.cos_lr,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Augmentation - tuned for medical imaging
        hsv_h=0.0,         # No hue augmentation (grayscale US)
        hsv_s=0.0,         # No saturation augmentation
        hsv_v=0.2,         # Mild brightness augmentation
        degrees=5.0,        # Slight rotation
        translate=0.1,      # Translation
        scale=0.3,          # Scale augmentation
        flipud=0.5,         # Vertical flip (ABUS can be flipped)
        fliplr=0.5,         # Horizontal flip
        mosaic=0.5,         # Reduced mosaic (medical images)
        mixup=0.1,          # Light mixup
        copy_paste=0.0,     # No copy-paste

        # Training behavior
        close_mosaic=15,    # Disable mosaic for final 15 epochs
        amp=True,           # Mixed precision
        cache="ram" if args.cache else False,
        rect=args.rect,

        # Loss weights
        box=7.5,            # Box loss weight
        cls=0.5,            # Classification loss weight
        dfl=1.5,            # Distribution focal loss weight

        # Validation
        val=True,
        iou=0.5,            # NMS IoU threshold for validation
        conf=0.001,         # Confidence threshold for validation

        # Save
        save=True,
        save_period=10,     # Save checkpoint every 10 epochs
        plots=True,
    )

    print("\n=== Training Configuration ===")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    print()

    # Start training
    results = model.train(**train_args)

    # Print final metrics
    print("\n=== Training Complete ===")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results directory: {model.trainer.save_dir}")

    # Run validation on best model
    print("\n=== Final Validation ===")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
