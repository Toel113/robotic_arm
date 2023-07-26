from logging import exception
import cv2
import pyrealsense2 as rs
import numpy as np
from scipy.spatial import distance as dist
import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from time import sleep

current_actual = None

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
           "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
           "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
           "SOFA", "TRAIN", "TVMONITOR"]

COLORS = np.random.uniform(0, 100, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt", "./MobileNetSSD/MobileNetSSD.caffemodel")

pipeline = rs.pipeline()
config = rs.config()

rs_w = 640
rs_h = 480
fps = 60

config.enable_stream(rs.stream.depth, rs_w, rs_h, rs.format.z16, fps)
config.enable_stream(rs.stream.color, rs_w, rs_h, rs.format.bgr8, fps)
pipeline.start(config)

cap = cv2.VideoCapture(2)


def connect_robot():
    try:
        ip = "192.168.1.6"  # "192.168.5.1"
        dashboard_p = 29999
        move_p = 30003
        feed_p = 30004
        print("Connecting...")
        dashboard = DobotApiDashboard(ip, dashboard_p)
        move = DobotApiMove(ip, move_p)
        feed = DobotApi(ip, feed_p)
        print("Connection succeeded.")
        return dashboard, move, feed
    except Exception as e:
        print("Connection failed.")
        raise e


def run_point_MOVL(move: DobotApiMove, point_list: list):
    # MovL(self, x, y, z, rx, ry, rz):
    move.MovL(point_list[0], point_list[1], point_list[2], point_list[3], point_list[4], point_list[5])


def run_point_MOVJ(move: DobotApiMove, point_list: list):
    # MovJ(self, x, y, z, rx, ry, rz):
    move.MovJ(point_list[0], point_list[1], point_list[2], point_list[3], point_list[4], point_list[5])


def get_feed(feed: DobotApi):
    global current_actual
    hasRead = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0

        a = np.frombuffer(data, dtype=MyType)
        if hex((a['test_value'][0])) == '0x123456789abcdef':
            # Refresh Properties
            # print("============== Feed Back ===============")
            current_actual = a["tool_vector_actual"][0]
            # print("tool_vector_actual: [X:{0}] , [Y:{1}] , [Z:{2}] , [RX:{3}] , [RY:{4}] , [RZ:{5}]".format(current_actual[0],current_actual[1],current_actual[2],current_actual[3],current_actual[4],current_actual[5]))

            CR_joint = a['q_target'][0]
            # print("CR_joint: [j1:{0}] , [j2:{1}] , [j3:{2}] , [j4:{3}] , [j5:{4}] , [j6:{5}]".format(CR_joint[0],CR_joint[1],CR_joint[2],CR_joint[3],CR_joint[4],CR_joint[5]))
            # print("========================================")
        sleep(0.001)


def wait_arrive(point_list):
    global current_actual
    while True:
        is_arrive = True
        if current_actual is not None:
            for index in range(len(current_actual)):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                return
        sleep(1)


if __name__ == '__main__':
    # Robot Setup
    dashboard, move, feed = connect_robot()
    print("Start power up.")
    dashboard.PowerOn()
    print("Please wait patiently,Robots are working hard to start.")
    count = 3
    while count > 0:
        print(count)
        count = count - 1
        sleep(1)

    # Enable Robot
    print("Clear error.")
    dashboard.ClearError()

    print("Start enable.")
    dashboard.EnableRobot()
    print("Complete enable.")

    try:
        while (True):
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
                        # z = depth_frame.get_distance(int(startX), int(startY))
                        # print("X: ", startX)
                        print("--------------------------")
                        print("CAMERA")
                        print("OBJ: ", CLASSES[class_index])
                        print("X: ", startX)
                        print("Y: ", startY)
                        # print("Z: ", z)

                        label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                        cv2.rectangle(frame, (startX - 1, startY - 30), (endX + 1, startY), COLORS[class_index], cv2.FILLED)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                        print("--------------------------------")
                        print("ROBOT")
                        x_robot = (startY + 45)
                        y_robot = (startX - 1004)
                        print("x", x_robot)
                        print("y", y_robot)
                        point = [x_robot, y_robot, 390, -177, 2, 128]
                        print(point)

                        run_point_MOVL(move, point)
                        sleep(2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


    except KeyboardInterrupt:
        dashboard.DisableRobot()
        print("Clear error.")
        dashboard.ClearError()