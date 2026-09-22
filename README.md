# Hệ CSDL lưu trữ và tìm kiếm tiếng động cơ máy nổ

Toàn bộ code bám sát Lecture 7 (Vector database & Clustering), Lecture 8
(Text indexing — dùng công thức Precision/Recall), Lecture 9 (Multimedia
metadata) và Lecture 10 (Indexing & Retrieval for Audio).

## Cài đặt môi trường ảo (khuyến nghị, tránh xung đột thư viện)

**Cách 1 — dùng script có sẵn:**
```bash
bash setup_env.sh
source venv/bin/activate        # Linux/Mac
# hoặc venv\Scripts\activate    # Windows
```

**Cách 2 — tạo thủ công:**
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# hoặc venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Từ đây, mọi lệnh `python3 ...` bên dưới đều chạy **trong** môi trường ảo
đã kích hoạt. Mở terminal mới thì phải `source venv/bin/activate` lại.

> **Lưu ý về phiên bản Python**: nên dùng **Python 3.10 – 3.12**. Trên
> Python 3.14, thư viện `llvmlite`/`numba` (mà `librosa` cần) không phát
> hành wheel dựng sẵn cho macOS x86_64 nên pip sẽ build từ mã nguồn và lỗi
> (thiếu LLVM). `setup_env.sh` đã tự chọn phiên bản Python tương thích.

## Thứ tự chạy

### 1. Tải 500 file theo danh sách (đáp ứng yêu cầu đề bài mục 1)

Danh sách 500 file cần dùng nằm trong `engine_sounds_500/manifest_500.csv`
(kèm nhãn `engine_set` = "loại động cơ", `state_label` = "trạng thái làm
việc", lấy trực tiếp từ ground-truth RPM/Torque của bộ dữ liệu — không cần
gán tay). Tải audio theo đúng danh sách này:

```bash
python3 select_and_download.py
```
→ tải về `engine_sounds_500/audio/<engine_set>__<filename>` cho từng dòng
trong manifest. File nào đã có sẵn sẽ được bỏ qua nên chạy lại rất nhanh.

Nếu `manifest_500.csv` chưa có, script sẽ tự sinh bằng stratified sampling
(đều theo 8 set × 5 mức RPM) rồi tải theo danh sách đó.

Cấu trúc sau khi tải:
```
engine_sounds_500/
├── audio/            # 500 file .wav, tên dạng <engine_set>__<filename>
└── manifest_500.csv  # engine_set, state_label, rpm_mean/std, torque_mean/std
```

### 2. Trích đặc trưng + xây CSDL (mục 2, 3)
```bash
python3 build_database.py --manifest engine_sounds_500/manifest_500.csv \
                            --audio_dir engine_sounds_500/audio \
                            --db engine_sounds.db
```
→ `feature_extraction.py` trích 39 chiều đặc trưng/file theo đúng công thức
Lecture 10 (E, ZCR, silence ratio, spectral centroid, bandwidth,
harmonicity, f0, MFCC-13, mỗi cái lấy mean+std → aggregation mức segment).
→ `database.py` lưu vào 3 bảng SQLite (`AudioFile`, `EngineMetadata`,
`FeatureVector`) theo mô hình phân lớp metadata Lecture 9.

### 3. Tìm kiếm tương đồng (mục 4a, 4b)
```python
from database import get_connection, load_all_vectors
from search import EngineSoundIndex
from feature_extraction import extract_feature_vector

conn = get_connection("engine_sounds.db")
ids, vectors, meta = load_all_vectors(conn)
index = EngineSoundIndex(vectors, meta)

q_vec, _ = extract_feature_vector("file_moi.wav")
top5 = index.query(q_vec, top_k=5, metric="cosine")   # dot product / cosine / L2 — Lecture 7
```
`search.py` cũng có `query_clustered()` dùng K-means (Lecture 7:
"Querying Clustered Vector Data") để chỉ quét các cluster gần nhất thay
vì toàn bộ CSDL — dùng để minh họa và so sánh tốc độ trong báo cáo.

### 4. Đánh giá (mục 4c)
```bash
python3 evaluate.py
```
Tính Precision@5 / Recall@5 theo đúng định nghĩa Lecture 8 (slide 11),
với "relevant" xác định tự động từ (cùng engine_set + rpm_mean gần nhau).
So sánh 3 cấu hình: full 39 đặc trưng, chỉ MFCC, cosine vs L2 — dùng script
này với `all_vectors, all_meta` thật lấy từ `load_all_vectors()` thay vì
dữ liệu tổng hợp trong `if __name__ == "__main__"`.

### 5. Demo (mục 5)
```bash
streamlit run app_demo.py -- --db engine_sounds.db
```
Giao diện: tải file WAV mới lên → xem bảng 39 đặc trưng đã trích (kết quả
trung gian) → xem top-5 kết quả kèm điểm tương đồng và nghe lại.

## Ghi chú khi viết báo cáo

- **Mục 1 (mô tả giống/khác)**: dùng `librosa.display.specshow` trên STFT
  (đã có trong `feature_extraction.stft_magnitude`) để vẽ spectrogram vài
  file mẫu — so sánh vị trí các dải hài (harmonic bands) giữa các
  `engine_set` khác nhau, và độ dày đặc năng lượng giữa các `state_label`
  khác nhau trong cùng 1 set.
- **Mục 2 (giá trị thông tin đặc trưng)**: đã giải thích trong docstring
  từng hàm ở `feature_extraction.py` — copy trực tiếp vào báo cáo.
- **Mục 4b (kết quả trung gian)**: chạy `evaluate.py`-style leave-one-out
  cho 3 file mẫu (cùng set, khác trạng thái / khác set) rồi in bảng
  vector đặc trưng + dot product + |D| + |Q| + cosine, đúng định dạng
  bài tập ví dụ ở Lecture 7 slide 10.
# HCSDLDPT
