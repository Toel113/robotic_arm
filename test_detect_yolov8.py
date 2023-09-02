from ultralytics import YOLO
import cv2

model = YOLO("yolov8s_custom.pt")

results = model(source=0, show=True, stream=True, conf=0.8)
for r in results:
    boxes = r.boxes
    x0 = r.boxes.xyxy
    print(x0)
    if cv2.waitKey(100) == ord('q'):
        break

results.release()
cv2.destroyAllWindows()