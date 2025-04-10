from read_serial import SerialReader
from face_reg_system import FaceRecognitionSystem
from send_signal_adafruit import AdafruitSignalSender

import time
reg_flag = False  # Biến để theo dõi trạng thái nhận tín hiệu
add_flag = False  # Biến để theo dõi trạng thái nhận tín hiệu
close_door = False  # Biến để theo dõi trạng thái nhận tín hiệu


def processing(data):
    global reg_flag,add_flag,close_door
    print("Dữ liệu nhận được:", data)
    if data == "REG":
        reg_flag = True
    elif data == "ADD":
        add_flag = True
    elif data == "CLOSE":
        close_door = True

sr = SerialReader(port='COM3', baudrate=115200)
sr.connect()

sender= AdafruitSignalSender(feed_name="door")

def OpenDoor():
    global reg_flag
    reg_flag = False
    sender.send_signal(1)
    print("Đã mở cửa")

def CloseDoor():
    global close_door
    close_door = False
    time.sleep(5)
    sender.send_signal(0)
    print("Đã đóng cửa")

face_reg_system = FaceRecognitionSystem(dataset_path="dataset", encodings_file="encodings.pickle")  

sr.start_reading(callback=processing)


while True:
    if reg_flag:
        print("Đang tiến hành đăng ký khuôn mặt...")
        face_reg_system.recognize_faces(on_recognized_callback=OpenDoor)
        time.sleep(1)
    if add_flag:
        print("Đang tiến hành thêm khuôn mặt...")
        face_reg_system.capture_images()
        face_reg_system.train_model()
        add_flag = False
    if close_door:
        CloseDoor()

    time.sleep(1)

