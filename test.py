from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import torch
import numpy as np
import pandas

model = YOLO("yolov8s_custom.pt")
model.classes = ['bottle']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    images = image.unsqueeze(0)

    results = model(source=0, show=True, stream=True, conf=0.7)
    for r in results:
        boxes = r.boxes
        masks = r.masks
        probs = r.probs
    detections = results.pandas().xyxy[0]
    for _, detection in detections.iterrows():
        x = detection[0]
        y = detection[1]
        print("Object detected: X={}, Y={}".format(x, y))

    cv2.imshow("Frame", frame)
    if cv2.waitKey(100) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
