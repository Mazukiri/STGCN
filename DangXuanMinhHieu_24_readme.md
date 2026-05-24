# THÔNG TIN DỰ ÁN BTL - NHÓM 24

## 1. Thành viên và Phân công công việc
- **Đặng Xuân Minh Hiếu (24002005) - Trưởng nhóm**:
    - Quản lý dự án, thu thập dữ liệu WLASL-100.
    - Phát triển giao diện Demo Real-time (GUI) sử dụng OpenCV.
    - Viết báo cáo và Slide.
- **Mai Hoàng Đức (22001565)**:
    - Thiết kế kiến trúc mạng ST-GCN cải tiến (Adaptive Graph).
    - Thực hiện huấn luyện và tối ưu hóa hyperparameter.
- **Lê Quý Công (22001550)**:
    - Tiền xử lý dữ liệu, trích xuất đặc trưng MediaPipe Holistic.
    - Xây dựng Baseline LSTM và thực hiện so sánh hiệu năng.

## 2. Hướng dẫn Dữ liệu
- **Link tải dữ liệu**: [WLASL Dataset (Kaggle/Official)](https://www.kaggle.com/datasets/risovv/wlasl-processed)
- **Cách thiết lập**:
    1. Tải file `WLASL_v0.3.json` và video gốc.
    2. Đặt vào thư mục `data/raw/`.
    3. Chạy `uv run python src/data_processing/01_process_dataset.py` để lọc WLASL-100.
    4. Chạy `uv run python src/data_processing/02_extract_landmarks.py` để trích xuất xương.

## 3. Tổ chức thư mục và Kịch bản thực nghiệm
### Cấu trúc thư mục:
- `src/models/`: Chứa định nghĩa mạng ST-GCN và LSTM.
- `src/data_processing/`: Scripts xử lý dữ liệu thô.
- `src/weights/`: Chứa trọng số mô hình đã huấn luyện (để chạy Demo ngay).
- `report/`: File LaTeX và PDF báo cáo.

### Kịch bản thực nghiệm:
1. **Huấn luyện**: `uv run python src/train_stgcn.py --epochs 100 --batch-size 32`
2. **Đánh giá**: `uv run python src/evaluate.py --model-path src/weights/best_stgcn_model.pth`
3. **Chạy Demo GUI**: `uv run python src/inference_webcam.py` (Yêu cầu Camera).
