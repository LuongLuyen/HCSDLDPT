"""
database.py
============
Xây dựng CSDL SQLite quản trị đặc trưng âm thanh, theo đúng mô hình
phân lớp metadata của Lecture 9 (Multimedia Metadata, slide 13-14:
Technical / Descriptive / Structural metadata) và cấu trúc bảng
"FeatureVector table" của Lecture 12 (Keyframe Index Structure, slide 18,
áp dụng tương tự cho audio thay vì video keyframe).

3 bảng:
  AudioFile      : technical metadata  (đường dẫn, thời lượng, sample rate...)
  EngineMetadata : descriptive metadata (loại động cơ = set A-H, trạng thái,
                   RPM/Torque ground-truth từ manifest_500.csv)
  FeatureVector  : vector index 39-chiều (Lecture 7: "vector space database")
"""

import sqlite3
import numpy as np
import json
from feature_extraction import FEATURE_NAMES

DB_PATH = "engine_sounds.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS AudioFile (
    audiooid    INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath    TEXT UNIQUE NOT NULL,
    filename    TEXT,
    duration_sec REAL,
    samplerate  INTEGER
);

CREATE TABLE IF NOT EXISTS EngineMetadata (
    audiooid    INTEGER PRIMARY KEY REFERENCES AudioFile(audiooid),
    engine_set  TEXT,        -- 'loại động cơ' (A_full_set .. H_large_set)
    state_label TEXT,        -- 'trạng thái làm việc' (idle_lowest_rpm .. very_high_rpm)
    rpm_mean    REAL,
    rpm_std     REAL,
    torque_mean REAL,
    torque_std  REAL
);

CREATE TABLE IF NOT EXISTS FeatureVector (
    audiooid    INTEGER PRIMARY KEY REFERENCES AudioFile(audiooid),
    vector_json TEXT NOT NULL   -- vector 39 chiều, lưu JSON để đọc lại bằng numpy
);
"""


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_audio_record(conn, filepath, filename, duration_sec, samplerate,
                         engine_set, state_label, rpm_mean, rpm_std,
                         torque_mean, torque_std, feature_vector):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO AudioFile (filepath, filename, duration_sec, samplerate) "
        "VALUES (?, ?, ?, ?)",
        (filepath, filename, duration_sec, samplerate),
    )
    cur.execute("SELECT audiooid FROM AudioFile WHERE filepath = ?", (filepath,))
    audiooid = cur.fetchone()[0]

    cur.execute(
        "INSERT OR REPLACE INTO EngineMetadata "
        "(audiooid, engine_set, state_label, rpm_mean, rpm_std, torque_mean, torque_std) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (audiooid, engine_set, state_label, rpm_mean, rpm_std, torque_mean, torque_std),
    )

    cur.execute(
        "INSERT OR REPLACE INTO FeatureVector (audiooid, vector_json) VALUES (?, ?)",
        (audiooid, json.dumps(feature_vector.tolist())),
    )
    conn.commit()
    return audiooid


def load_all_vectors(conn):
    """
    Trả về: audiooids (list), matrix (n_files x 39 numpy array), metadata (list of dict)
    -- đây chính là bảng dữ liệu dùng cho vector-space query ở Lecture 7
    (Di = [Ti1..TiN], slide 4).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT f.audiooid, f.filepath, f.filename,
               m.engine_set, m.state_label, m.rpm_mean, m.torque_mean,
               v.vector_json
        FROM AudioFile f
        JOIN EngineMetadata m ON f.audiooid = m.audiooid
        JOIN FeatureVector v ON f.audiooid = v.audiooid
    """)
    rows = cur.fetchall()

    audiooids, vectors, meta = [], [], []
    for r in rows:
        audiooid, filepath, filename, engine_set, state_label, rpm_mean, torque_mean, vjson = r
        audiooids.append(audiooid)
        vectors.append(json.loads(vjson))
        meta.append({
            "audiooid": audiooid, "filepath": filepath, "filename": filename,
            "engine_set": engine_set, "state_label": state_label,
            "rpm_mean": rpm_mean, "torque_mean": torque_mean,
        })
    return audiooids, np.array(vectors), meta


if __name__ == "__main__":
    conn = get_connection("/tmp/test.db")
    import soundfile as sf
    dummy_vec = np.random.randn(len(FEATURE_NAMES))
    insert_audio_record(
        conn, "/tmp/dummy.wav", "dummy.wav", 3.0, 22050,
        "A_full_set", "mid", 3000.0, 200.0, 150.0, 20.0, dummy_vec,
    )
    ids, mat, meta = load_all_vectors(conn)
    print("Số bản ghi:", len(ids), "| shape vector matrix:", mat.shape)
    print("Metadata mẫu:", meta[0])
