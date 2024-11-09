from ultralytics import YOLO
from tests import SOURCE, MODEL


def test_export_tflite():
    """Test YOLO exports to TFLite format under specific OS and Python version conditions."""
    model = YOLO(MODEL)
    file = model.export(format="tflite", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


def test_export_imx500_ptq():
    """Test YOLOv8n exports to imx500 format."""
    model = YOLO("yolov8n.pt")
    file = model.export(format="imx500", imgsz=32, gptq=False)
    YOLO(file)(SOURCE, imgsz=32)
