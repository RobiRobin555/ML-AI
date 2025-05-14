
# Hướng dẫn chạy file `BTL4.py`

## 📋 Mô tả
File `BTL4.py` thực hiện phân tích và huấn luyện mô hình phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection) bằng thuật toán Random Forest, có xử lý mất cân bằng dữ liệu bằng SMOTE và tối ưu mô hình với GridSearchCV.

---

## 🗂 Yêu cầu hệ thống

### 1. Python
- Python 3.8 hoặc mới hơn

### 2. Thư viện cần cài đặt

Bạn cần cài đặt các thư viện sau trước khi chạy chương trình:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

---

## 📁 Cấu trúc thư mục

```
├── BTL4.py
├── creditcard_2023.csv   # Dữ liệu đầu vào (phải nằm cùng thư mục với file .py)
└── README.md
```

---

## ▶️ Cách chạy chương trình

Bước 1: Đảm bảo rằng file `creditcard_2023.csv` đã được đặt trong cùng thư mục với `BTL4.py`.

Bước 2: Mở terminal hoặc command prompt, di chuyển đến thư mục chứa file và chạy lệnh sau:

```bash
python BTL4.py
```

---

## ✅ Các bước xử lý trong chương trình

1. Đọc dữ liệu từ file `creditcard_2023.csv`
2. Lấy mẫu 20% dữ liệu để tăng tốc xử lý
3. Chuẩn hóa cột `Amount` bằng StandardScaler
4. Xử lý mất cân bằng lớp bằng SMOTE
5. Chia dữ liệu thành tập huấn luyện và kiểm thử
6. Huấn luyện mô hình Random Forest
7. Đánh giá mô hình bằng accuracy, classification report, confusion matrix
8. Tối ưu mô hình với GridSearchCV

---

## 🖼 Kết quả

- **Accuracy**
- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: được hiển thị bằng biểu đồ
- **Best Parameters**: tham số tối ưu từ GridSearchCV

---

## 📝 Ghi chú

- Nếu bạn dùng Jupyter Notebook, có thể copy từng khối lệnh từ file `.py` vào từng cell để chạy tuần tự và quan sát kết quả.
- Chương trình có sử dụng hiển thị đồ thị (`matplotlib` + `seaborn`) nên cần có môi trường hỗ trợ GUI nếu chạy bằng terminal.
