# Custom FCOS Object Detector

Dự án này triển khai một mô hình phát hiện đối tượng đa lớp tùy chỉnh (Custom Object Detector) từ đầu dựa trên kiến trúc **FCOS (Fully Convolutional One-Stage Object Detection)** kết hợp mạng trích xuất đặc trưng **ResNet-50 Pretrained** và **FPN (Feature Pyramid Network)**. Mô hình sử dụng hàm mất mát **Smooth L1 Loss** cho định vị và **Sigmoid Focal Loss** cho phân loại, kết hợp với các kỹ thuật tăng cường dữ liệu mạnh mẽ để đạt điểm số mAP cao nhất.

---

## 1. Cấu trúc thư mục (Directory Structure)

Thư mục nộp bài được cấu trúc như sau:
```
<my_submission>/
├── models/
│   ├── __init__.py
│   ├── detector.py      # Định nghĩa mô hình FCOS + FPN + Shared Head
│   └── loss.py          # Hàm mất mát Focal Loss, Smooth L1 Loss & Centerness Loss
├── utils/
│   ├── __init__.py
│   ├── dataset.py       # Bộ đọc dữ liệu & các kỹ thuật Tăng cường (Augmentations)
│   └── od_utils.py      # NMS viết từ đầu, Target Assignment, chuyển đổi tọa độ
├── train.py             # Script quản lý quy trình huấn luyện & đánh giá mAP
├── predict.py           # Script suy luận đầu ra định dạng JSON
├── README.md            # Tài liệu hướng dẫn sử dụng này
└── requirements.txt     # Danh sách thư viện yêu cầu
```

---


## 2. Quy trình huấn luyện (Training Procedure)

Chạy lệnh huấn luyện bắt buộc (hỗ trợ tự động tính toán mAP sau mỗi epoch để lưu checkpoint tốt nhất vào `./models/best.pth`):

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --multi_scale
```

*Các tham số tùy chọn bổ sung:*
*   `--epochs`: Số epoch huấn luyện (mặc định: `30`).
*   `--batch_size`: Kích thước batch (mặc định: `8`, có thể giảm xuống `4` nếu thiếu VRAM).
*   `--lr`: Tốc độ học tập khởi tạo (mặc định: `1e-4`).
*   `--multi_scale`: Bật tính năng huấn luyện đa kích thước (Multi-scale training) để tăng mAP đối với các vật thể nhỏ/lớn.
*   `--target_size`: Độ phân giải ảnh đầu vào (mặc định: `512`).
*   `--eval_interval`: Chu kỳ đánh giá trên tập Validation (mặc định: `5` epoch một lần).
*   `--patience`: Số lần đánh giá không cải thiện mAP liên tiếp trước khi kích hoạt dừng sớm - Early Stopping (mặc định: `3` lần đánh giá, tương đương `15` epoch).

---

## 3. Quy trình suy luận (Inference/Prediction Procedure)

Chạy lệnh suy luận để dự đoán nhãn và hộp bao từ một thư mục ảnh bất kỳ:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json \
  --conf_thresh 0.1 \
  --nms_thresh 0.5
```

*Các tham số tùy chọn bổ sung:*
*   `--model_path`: Đường dẫn tới tệp checkpoint trọng số tốt nhất (mặc định: `./models/best.pth`).
*   `--conf_thresh`: Ngưỡng độ tin cậy để lọc các hộp bao dự đoán (mặc định: `0.1`).
*   `--nms_thresh`: Ngưỡng IoU cho thuật toán NMS (mặc định: `0.5`).

---

## 4. Tự chấm điểm thử nghiệm (Self-Evaluation)


```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```
