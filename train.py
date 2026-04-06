from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(
        data=r"E:\Dalnex\Bottle.v3-0.1.1.yolov8\data.yaml",
        task="detect",
        epochs=80,
        imgsz=640,
        batch=8,
        name="bottle_detector_v3",
        workers=2,

        # augmentation
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
        degrees=5, translate=0.1, scale=0.2,
        fliplr=0.5, mosaic=0.5, mixup=0.0,

        # loss (defaults — safe for single class)
        cls=0.5, box=7.5,

        optimizer="SGD", lr0=0.01,
        patience=20, iou=0.5,
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()