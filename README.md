# 🚦 HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG SỬ DỤNG YOLOv8

## 📌 Giới thiệu

Đây là dự án xây dựng hệ thống nhận diện biển báo giao thông sử dụng mô hình học sâu YOLOv8. Hệ thống cho phép người dùng tải ảnh biển báo giao thông lên giao diện web, sau đó tự động phát hiện vị trí, phân loại biển báo và hiển thị kết quả nhận diện cùng độ tin cậy.

Dự án được phát triển nhằm mục đích học tập, nghiên cứu và ứng dụng công nghệ Trí tuệ nhân tạo (AI) trong lĩnh vực thị giác máy tính (Computer Vision).

---

## 🎯 Mục tiêu đề tài

* Xây dựng mô hình nhận diện biển báo giao thông bằng YOLOv8.
* Phát hiện chính xác vị trí biển báo trong ảnh.
* Phân loại các loại biển báo giao thông.
* Hiển thị khung nhận diện (Bounding Box).
* Hiển thị độ tin cậy của kết quả dự đoán.
* Xây dựng giao diện web trực quan bằng Streamlit.

---

## 🛠 Công nghệ sử dụng

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit
* NumPy

---

## 📂 Cấu trúc thư mục

```text
traffic-sign-recognition/
│
├── app.py
├── train_yolo.py
├── test_model.py
├── yolov8n.pt
│
├── traffic sign.v8i.yolov8/
│   ├── train/
│   ├── valid/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
└── runs/
```

---

## 🚦 Các biển báo được hỗ trợ

Mô hình hiện có khả năng nhận diện:

* Biển giới hạn tốc độ 40 km/h
* Biển giới hạn tốc độ 60 km/h
* Biển giới hạn tốc độ 80 km/h
* Biển giới hạn tốc độ 100 km/h
* Biển giới hạn tốc độ 120 km/h
* Biển cấm đi thẳng
* Biển cấm rẽ trái
* Biển cấm rẽ phải
* Biển cấm quay đầu bên trái
* Biển cấm quay đầu bên phải

---

## 🧠 Huấn luyện mô hình

Mô hình được huấn luyện bằng YOLOv8 với tập dữ liệu biển báo giao thông.

Ví dụ đoạn mã huấn luyện:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="traffic sign.v8i.yolov8/data.yaml",
    epochs=40,
    imgsz=416,
    batch=4,
    workers=0
)
```

Sau khi huấn luyện, mô hình tốt nhất được lưu tại:

```text
runs/detect/train-6/weights/best.pt
```

---

## ▶️ Hướng dẫn cài đặt

### Cài đặt thư viện

```bash
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install numpy
```

### Chạy hệ thống

```bash
streamlit run app.py
```

---

## 📷 Quy trình hoạt động của hệ thống

1. Người dùng tải ảnh biển báo giao thông lên giao diện web.
2. Hệ thống đọc ảnh bằng OpenCV.
3. Mô hình YOLOv8 thực hiện phát hiện đối tượng.
4. Xác định vị trí biển báo bằng Bounding Box.
5. Phân loại loại biển báo.
6. Tính toán độ tin cậy của dự đoán.
7. Hiển thị kết quả nhận diện trên giao diện web.

---

## 📊 Kết quả đầu ra

Hệ thống cung cấp:

* Ảnh gốc.
* Ảnh sau khi nhận diện.
* Loại biển báo giao thông.
* Độ tin cậy của mô hình.
* Bounding Box khoanh vùng biển báo.

Ví dụ:

```text
Loại biển báo: Biển giới hạn tốc độ 100 km/h
Độ tin cậy: 91.7%
```

---

## 👨‍💻 Tác giả

**Nguyễn Đức Hoàn**

Sinh viên Khoa Công nghệ Thông tin

Trường Đại học Đại Nam

---

## 📜 Mục đích sử dụng

Dự án được xây dựng phục vụ học tập, nghiên cứu và thực hành các kỹ thuật xử lý ảnh, thị giác máy tính và trí tuệ nhân tạo.
