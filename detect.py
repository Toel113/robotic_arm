import numpy as np
import cv2
import pyrealsense2 as rs


CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]

COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")

pipeline= rs.pipeline()
config= rs.config()

rs_w=640
rs_h=480
fps=60

config.enable_stream(rs.stream.depth, rs_w, rs_h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, rs_w, rs_h, rs.format.bgr8, fps)
pipeline.start(config)

cap = cv2.VideoCapture(1)



while True:
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	ret, frame = cap.read()
	if ret:
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			percent = detections[0, 0, i, 2]
			if percent > 0.8:
				class_index = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				z_cam = depth_frame.get_distance(int(startX), int(startY))
				# print("X: ", startX)
				print("--------------------------")
				print("CAMERA")
				print("OBJ: ", CLASSES[class_index])
				print("X: ", startX)
				print("Y: ", startY)
				print("Z: ", z_cam)

				label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
				cv2.rectangle(frame, (startX - 1, startY - 30), (endX + 1, startY), COLORS[class_index], cv2.FILLED)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


				# print("--------------------------------")
				# print("ROBOT")
				# x_robot = (startY + 125)
				# y_robot = (startX - 185)
				# z_robot = (z_cam * 100)
				# print(x_robot)
				# print(y_robot)
				# print(z_robot)
				# point = [x_robot, y_robot, z_robot, -179, 0, -90]
				# print(point)

		cv2.imshow("Frame", frame)
		if cv2.waitKey(100) & 0xFF==ord('q'):
			break


cap.release()
cv2.destroyAllWindows()