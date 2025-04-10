import serial
import threading

class SerialReader:
    def __init__(self, port, baudrate=9600, timeout=3):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.running = False
        self.thread = None

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Kết nối thành công với {self.port} ở baudrate {self.baudrate}")
        except serial.SerialException as e:
            print(f"Lỗi khi kết nối đến Serial: {e}")

    def start_reading(self, callback=None):
        if self.serial and self.serial.is_open:
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, args=(callback,))
            self.thread.start()
        else:
            print("Chưa kết nối hoặc cổng không mở.")

    def _read_loop(self, callback):
        while self.running:
            if self.serial.in_waiting:
                try:
                    data = self.serial.readline().decode('utf-8').strip()
                    print(f"Nhận: {data}")
                    if callback:
                        callback(data)
                except Exception as e:
                    print(f"Lỗi khi đọc dữ liệu: {e}")

    def stop_reading(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Đã ngắt kết nối Serial.")

