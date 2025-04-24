import os
import cv2
import face_recognition

class FaceExtractor:
    def __init__(self, dataset_path=r"C:\Users\Divu\Desktop\DADN\detect_face\dataset", output_dir="extracted_faces"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.detection_method = "hog"  # Có thể đổi thành "cnn" nếu dùng GPU

    def draw_rectangles(self, face_img):
        """Vẽ khung hình chữ nhật quanh khuôn mặt trên vùng đã cắt."""
        padding = 20  # Số pixel mở rộng mỗi cạnh
        top_padded = padding
        right_padded = face_img.shape[1] - padding
        bottom_padded = face_img.shape[0] - padding
        left_padded = padding
        cv2.rectangle(face_img, (left_padded, top_padded), (right_padded, bottom_padded), (0, 255, 0), 2)
        cv2.putText(face_img, "Face", (left_padded, top_padded - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return face_img

    def extract_faces(self):
        """Cắt khuôn mặt từ ảnh trong dataset và lưu theo cấu trúc, giữ cấu trúc thư mục."""
        print(f"🚀 Bắt đầu trích xuất khuôn mặt từ dataset: {self.dataset_path}")
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

        # Lấy danh sách các thư mục con (mỗi thư mục là một người/nhãn)
        subdirs = [d for d in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not subdirs:
            print("❌ Không tìm thấy thư mục con nào trong dataset")
            return

        total_face_count = 0
        image_extensions = (".jpg", ".jpeg", ".png")

        for subdir in subdirs:
            input_subdir = os.path.join(self.dataset_path, subdir)
            output_subdir = os.path.join(self.output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            
            print(f"📂 Xử lý thư mục: {subdir}")
            
            # Lấy danh sách ảnh trong thư mục con
            image_paths = [os.path.join(input_subdir, f) for f in os.listdir(input_subdir) 
                          if f.lower().endswith(image_extensions)]
            
            face_count = 0
            for idx, image_path in enumerate(image_paths):
                print(f"📸 Xử lý ảnh {idx + 1}/{len(image_paths)}: {image_path}")
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ Không thể đọc ảnh: {image_path}")
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image, model=self.detection_method)

                if not face_locations:
                    print(f"⚠️ Không phát hiện khuôn mặt trong: {image_path}")
                    continue

                for face_idx, (top, right, bottom, left) in enumerate(face_locations):
                    padding = 30
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    right = min(image.shape[1], right + padding)
                    bottom = min(image.shape[0], bottom + padding)

                    face_img = image[top:bottom, left:right]
                    
                    if face_img.size == 0:
                        print(f"⚠️ Vùng khuôn mặt không hợp lệ trong: {image_path}")
                        continue

                    filename = f"face_{idx:05d}_{face_idx}.png"
                    filepath = os.path.join(output_subdir, filename)
                    cv2.imwrite(filepath, face_img)
                    print(f"✅ Đã lưu khuôn mặt: {filepath}")
                    face_count += 1

            print(f"✅ Hoàn tất thư mục {subdir}: {face_count} khuôn mặt")
            total_face_count += face_count

        print(f"✅ Hoàn tất toàn bộ dataset! Đã lưu {total_face_count} khuôn mặt vào {self.output_dir}")

if __name__ == "__main__":
    # Nhập đường dẫn đến dataset

    extractor = FaceExtractor( )
    extractor.extract_faces()