import os
import cv2
import time
import pickle
import face_recognition
from datetime import datetime
from imutils import paths
import imutils


class FaceRecognitionSystem:
    def __init__(self, dataset_path="dataset", encodings_file="encodings.pickle", detection_method="hog"):
        self.dataset_path = dataset_path
        self.encodings_file = encodings_file
        self.detection_method = detection_method

    def capture_images(self, num_images=10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_folder = f"temp_{timestamp}"
        output_dir = os.path.join(self.dataset_path, temp_folder)

        os.makedirs(output_dir, exist_ok=True)

        video = cv2.VideoCapture(0)
        images = []
        print(f"🚀 Bắt đầu chụp {num_images} ảnh...")

        count = 0
        while count < num_images:
            ret, frame = video.read()
            if not ret:
                print("❌ Không thể lấy hình từ camera.")
                break

            cv2.imshow("Video", frame)
            images.append(frame)
            print(f"📸 Đã chụp ảnh thứ {count + 1}/{num_images}")
            count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1)

        video.release()
        cv2.destroyAllWindows()

        for idx, img in enumerate(images):
            filename = str(idx).zfill(5) + ".png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)

    def train_model(self):
        print("🔧 Bắt đầu huấn luyện dữ liệu khuôn mặt...")
        folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]

        all_encodings = []
        all_names = []

        for folder in folders:
            if folder == "unknown":
                continue
            path_to_images = os.path.join(self.dataset_path, folder)
            image_paths = list(paths.list_images(path_to_images))
            print(f"📂 Huấn luyện từ thư mục: {folder} ({len(image_paths)} ảnh)")

            for i, image_path in enumerate(image_paths):
                print(f"➡️ Ảnh {i + 1}/{len(image_paths)}: {image_path}")
                image = cv2.imread(image_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model=self.detection_method)
                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    all_encodings.append(encoding)
                    all_names.append(folder)

        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                existing_data = pickle.load(f)
            existing_data["encodings"].extend(all_encodings)
            existing_data["names"].extend(all_names)
            data = existing_data
        else:
            data = {"encodings": all_encodings, "names": all_names}

        with open(self.encodings_file, "wb") as f:
            pickle.dump(data, f)
        print("✅ Huấn luyện xong và đã lưu vào:", self.encodings_file)

    def recognize_faces(self, on_recognized_callback=None):
        print("📹 Đang mở camera để nhận diện khuôn mặt...")

        with open(self.encodings_file, "rb") as f:
            data = pickle.load(f)

        video = cv2.VideoCapture(0)
        time.sleep(2.0)

        recognized_sequence = []
        unknown_dir = os.path.join(self.dataset_path, "unknown")
        os.makedirs(unknown_dir, exist_ok=True)
        count =0 
        max_count = 20
        while count<max_count:
            count += 1
            ret, frame = video.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(rgb, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=self.detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)

            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
                name = "Unknown"

                if True in matches:
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matched_idxs:
                        matched_name = data["names"][i]
                        counts[matched_name] = counts.get(matched_name, 0) + 1
                    name = max(counts, key=counts.get)

                names.append(name)

            if len(names) == 1:
                recognized_sequence.append(names[0])
            else:
                recognized_sequence.append("Unknown")

            # Nếu đã nhận diện 3 ảnh
            if len(recognized_sequence) == 3:
                if recognized_sequence[0] == recognized_sequence[1] == recognized_sequence[2] and recognized_sequence[0] != "Unknown":
                    print(f"✅ Phát hiện cùng một người: {recognized_sequence[0]}")
                    if on_recognized_callback:
                        on_recognized_callback()
                    break
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"unknown_{timestamp}.png"
                    filepath = os.path.join(unknown_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print("❌ Không khớp người. Đã lưu vào unknown.")
                recognized_sequence = []

            # Hiển thị camera
            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()
        print("🛑 Đã dừng nhận diện.")


# 👉 Sử dụng:
if __name__ == "__main__":
    frs = FaceRecognitionSystem(dataset_path=r"C:\Users\Divu\Desktop\DADN\detect_face\extracted_faces")
    #frs.train_model()
    frs.recognize_faces()
    