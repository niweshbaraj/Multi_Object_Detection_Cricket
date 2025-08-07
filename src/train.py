from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolov8m.pt')  # Use yolov8n.pt for speed, yolov8m.pt for higher accuracy
    model.train(data='data/annotations/cricket.yaml', epochs=50, imgsz=640, batch=8, name='yolov8-cricket')

if __name__ == '__main__':
    train_yolo()