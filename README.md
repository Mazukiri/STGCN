# WLASL - Word-Level Sign Language Recognition

Dự án nhận diện ngôn ngữ ký hiệu dựa trên bộ dữ liệu WLASL-Processed. Đồ án đề xuất giải pháp sử dụng **MediaPipe Holistic** để trích xuất khung xương 3D và mạng **ST-GCN** để nhận diện 100 từ vựng phổ biến nhất.

## 1. Thông tin thành viên & Phân công công việc

| STT | Họ và Tên | MSSV | Phân công công việc |
|---|---|---|---|
| 1 | **Mai Hoàng Đức** | 22001565 | Tìm hiểu lý thuyết ST-GCN, thiết kế kiến trúc Model `stgcn.py`, tinh chỉnh Hyperparameters và viết các script huấn luyện mô hình. |
| 2 | **Lê Quý Công** | 22001550 | Phụ trách pipeline xử lý dữ liệu WLASL, trích xuất MediaPipe Landmarks, và áp dụng thuật toán chuẩn hóa không gian (Spatial Normalization). |
| 3 | **Đặng Xuân Minh Hiếu** | 24002005 | Cài đặt ứng dụng `inference_webcam.py` chạy Real-time, phân tích EDA (Charts/Plots), viết Báo cáo LaTeX & thiết kế Slide. |

> **Giảng viên hướng dẫn:** TS. Cao Văn Chung

---

## 2. Hướng dẫn tải Dữ liệu (WLASL)

Bộ dữ liệu gốc cần được tải về và đặt đúng vị trí để các kịch bản có thể hoạt động.

1. Tải bộ dữ liệu gốc từ: [Kaggle WLASL Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) hoặc [WLASL Official GitHub](https://github.com/dxli94/WLASL).
2. Tải file `WLASL_v0.3.json` (chứa metadata và bounding box) và các tệp video MP4 tương ứng.
3. Giải nén và đặt dữ liệu vào thư mục `data/raw/` theo cấu trúc:
   - `data/raw/WLASL_v0.3.json`
   - `data/raw/videos/` (chứa toàn bộ file `.mp4`)

---

## 3. Cấu trúc thư mục (Folder Structure)

```text
CV Final Project/
│
├── data/
│   ├── raw/                    # Chứa WLASL_v0.3.json và thư mục videos/
│   └── processed/              # Nơi lưu trữ metadata đã lọc và tọa độ landmarks (.npy)
│
├── report/                     # Toàn bộ Báo cáo LaTeX (main.tex) và Slide (slides.tex)
│   ├── images/                 # Ảnh tĩnh tĩnh minh họa cho báo cáo
│   ├── main.pdf                # Báo cáo hoàn chỉnh
│   └── slides.pdf              # Slide thuyết trình hoàn chỉnh
│
├── src/                        # Mã nguồn dự án
│   ├── data_processing/        # Kịch bản tiền xử lý dữ liệu
│   │   ├── 01_process_dataset.py
│   │   └── 02_extract_landmarks.py
│   │
│   ├── models/                 # Chứa các định nghĩa mạng (stgcn.py, graph.py)
│   ├── weights/                # Thư mục lưu checkpoint trọng số mô hình
│   │
│   ├── dataset.py              # Dataloader của PyTorch
│   ├── train_stgcn.py          # Script huấn luyện chính cho ST-GCN
│   ├── evaluate.py             # Script đánh giá trên tập Test (Precision, Recall, F1)
│   ├── inference_webcam.py     # Ứng dụng chạy nhận diện trực tiếp bằng Camera
│   └── generate_report_charts.py # Script sinh hình ảnh biểu đồ tự động
│
├── requirements.txt            # Danh sách các thư viện Python
└── README.md                   # File hướng dẫn (bạn đang đọc)
```

---

## 4. Kịch bản Thực nghiệm (How to Run)

### Bước 0: Cài đặt môi trường
Chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 1: Tiền xử lý dữ liệu (Data Preprocessing)
Đảm bảo dữ liệu thô đã nằm trong `data/raw/`. Sau đó chạy lần lượt:

1. Xử lý metadata và lọc 100 classes:
```bash
python src/data_processing/01_process_dataset.py
```
2. Trích xuất đặc trưng Khung xương bằng MediaPipe (quá trình này có thể mất thời gian tùy vào CPU):
```bash
python src/data_processing/02_extract_landmarks.py
```
*Kết quả sẽ được lưu dưới dạng file `.npy` tại thư mục `data/processed/`.*

### Bước 2: Huấn luyện Mô hình ST-GCN
Để huấn luyện mô hình ST-GCN với các thiết lập mặc định, chạy lệnh:
```bash
python src/train_stgcn.py
```
*Trọng số của epoch tốt nhất sẽ được tự động lưu vào thư mục `src/weights/`.*

### Bước 3: Đánh giá Mô hình (Evaluation)
Đánh giá độ chính xác của mô hình trên tập Test:
```bash
python src/evaluate.py
```

### Bước 4: Ứng dụng Nhận diện Thời gian thực (Real-time Webcam)
Khởi động Camera, thực hiện ngôn ngữ ký hiệu bằng tay để xem hệ thống dự đoán trực tiếp (100 classes):
```bash
python src/inference_webcam.py
```
*(Bấm phím `Q` để tắt ứng dụng Camera).*
