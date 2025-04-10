from Adafruit_IO import Client
import random
import time
import sys
from key import ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY, FEED_NAME

class AdafruitSignalSender:
    def __init__(self, feed_name, usernamefunction=ADAFRUIT_IO_USERNAME, key=ADAFRUIT_IO_KEY):
        self.client = Client(usernamefunction, key)
        self.feed_name = feed_name

    def send_signal(self, signal: int = 0):
        try:
            self.client.send_data(self.feed_name, signal)
            print(f"Đã gửi tín hiệu: {signal} đến feed: {self.feed_name}")
        except Exception as e:
            print(f"Lỗi khi gửi tín hiệu: {e}")


# if __name__ == "__main__":
#     # Tạo đối tượng AdafruitSignalSender
#     sender = AdafruitSignalSender(feed_name=FEED_NAME)

#     # Gửi tín hiệu 1 đến feed "door"
#     sender.send_signal(1)
#     time.sleep(1)  # Đợi 1 giây trước khi gửi tín hiệu tiếp theo

#     # Gửi tín hiệu 0 đến feed "door"
#     sender.send_signal(0)