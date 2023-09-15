from ultralytics import YOLO


def main():
    """training was done in google colab instead."""

    model_type = "yolov8s-pose.pt"

    model = YOLO(model_type)

    results = model.train(data="config.yaml", epochs=200, imgsz=640)


if __name__ == "__main__":
    main()
