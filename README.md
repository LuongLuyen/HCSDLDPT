# Hệ CSDL lưu trữ và tìm kiếm tiếng động cơ máy nổ

Tìm kiếm tương đồng theo nội dung audio: đưa vào 1 file tiếng động cơ,
trả về 5 file giống nhất. Dataset 500 file WAV (8 nhóm A–H) + CSDL SQLite
đi kèm sẵn — không cần tải thêm.

## Chạy trên Windows

**1. Cài đặt 1 lần** (PowerShell tại thư mục này, Python 3.10–3.13):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Chạy demo:** nhấn đúp `ChayDemo.bat`, mở `http://localhost:8501`.

Tải file WAV lên → xem bảng 39 đặc trưng → xem Top-5 + nghe lại.

## Kiểm tra nhanh

```powershell
.\venv\Scripts\Activate.ps1
python -c "from database import get_connection, load_all_vectors; c=get_connection('engine_sounds.db'); ids,v,m=load_all_vectors(c); print(len(ids), v.shape)"
# phải ra: 500 (500, 39)
python evaluate.py   # Precision@5 / Recall@5 (tham khảo: 0,410 / 0,369 / 0,386)
```

## File chính

| File | Nội dung |
| ---- | -------- |
| `dataset/audio/` | 500 file WAV, tên `NHOM_Ten_STT.wav` (STT 001–500) |
| `dataset/DanhSach.csv` | `stt, ten_file, ten_chi_tiet` (nhóm xem ở đầu tên file) |
| `engine_sounds.db` | 3 bảng `AudioFile`, `EngineMetadata`, `FeatureVector` |
| `feature_extraction.py` | Trích vector 39 chiều mỗi file |
| `search.py` | Tìm kiếm cosine / dot / L2 + K-means |
| `evaluate.py` | Đánh giá Precision@5 / Recall@5 |
| `app_demo.py` | Giao diện Streamlit |

Sự cố: báo `running scripts is disabled` → chạy
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
rồi activate lại. Port 8501 bận → thêm `--server.port 8502` vào lệnh demo.
