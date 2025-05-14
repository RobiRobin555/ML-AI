# Import thư viện
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Bước 1: Đọc và lấy mẫu dữ liệu
df = pd.read_csv('creditcard_2023.csv')

# Chọn mẫu 20% dữ liệu để giảm thời gian chạy (tỉ lệ lớp không thay đổi)
df_sample = df.sample(frac=0.2, random_state=42)

print(df.info())
print(df.describe())
# Bước 2: Kiểm tra và xử lý dữ liệu bị thiếu
print("Số lượng giá trị bị thiếu:")
print(df_sample.isnull().sum())

# Bước 3: Chuẩn hóa cột Amount
scaler = StandardScaler()
df_sample['Amount'] = scaler.fit_transform(df_sample[['Amount']])



# Bước 5: Xử lý mất cân bằng dữ liệu bằng SMOTE
X = df_sample.drop(columns=['Class'], axis=1)
y = df_sample['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Kiểm tra lại phân phối dữ liệu sau SMOTE
print("Phân phối dữ liệu sau SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Bước 6: Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Bước 7: Huấn luyện mô hình Random Forest
# Khởi tạo mô hình
rf_model = RandomForestClassifier(
    n_estimators=100,       # Số lượng cây
    max_depth=7,            # Độ sâu tối đa của mỗi cây
    min_samples_split=5,    # Số mẫu tối thiểu để chia nhánh
    min_samples_leaf=2,     # Số mẫu tối thiểu tại mỗi lá
    class_weight="balanced",# Cân bằng trọng số cho các lớp
    random_state=42,
    n_jobs=-1               # Sử dụng tất cả các CPU để tăng tốc
)

# Huấn luyện mô hình
rf_model.fit(X_train, y_train)

# Bước 8: Đánh giá mô hình trên tập kiểm thử
y_pred = rf_model.predict(X_test)

# Đánh giá hiệu suất
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Hiển thị Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bước 9: Tối ưu hóa mô hình bằng GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,  # Giảm số lần cross-validation để tăng tốc
    scoring='f1',
    verbose=1
)

grid_search.fit(X_train, y_train)

# Hiển thị tham số tốt nhất
print(f"Best Parameters: {grid_search.best_params_}")

