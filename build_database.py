"""
build_database.py
==================
Chạy 1 lần để trích đặc trưng cho toàn bộ 500 file (thư mục engine_sounds_500/
tạo bởi select_and_download.py ở bước trước) và nạp vào CSDL SQLite.

Cách dùng:
    python3 build_database.py \
        --manifest engine_sounds_500/manifest_500.csv \
        --audio_dir engine_sounds_500/audio \
        --db engine_sounds.db
"""

import argparse
import os
import time
import pandas as pd
import soundfile as sf

from feature_extraction import extract_feature_vector
from database import get_connection, insert_audio_record


def main(manifest_path, audio_dir, db_path):
    manifest = pd.read_csv(manifest_path)
    conn = get_connection(db_path)

    n_ok, n_fail = 0, 0
    t0 = time.time()

    for i, row in manifest.iterrows():
        set_name = row["engine_set"]
        fname = row["filename"]
        local_name = f"{set_name}__{fname}"
        path = os.path.join(audio_dir, local_name)

        if not os.path.exists(path):
            print(f"[bỏ qua] không tìm thấy file: {path}")
            n_fail += 1
            continue

        try:
            vec, extra = extract_feature_vector(path)
            info = sf.info(path)

            insert_audio_record(
                conn,
                filepath=path,
                filename=local_name,
                duration_sec=info.duration,
                samplerate=info.samplerate,
                engine_set=set_name,
                state_label=row.get("state_label", None),
                rpm_mean=row.get("rpm_mean", extra.get("rpm_mean_from_channel")),
                rpm_std=row.get("rpm_std", None),
                torque_mean=row.get("torque_mean", extra.get("torque_mean_from_channel")),
                torque_std=row.get("torque_std", None),
                feature_vector=vec,
            )
            n_ok += 1
        except Exception as e:
            print(f"[lỗi] {local_name}: {e}")
            n_fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... đã xử lý {i+1}/{len(manifest)} file ({elapsed:.1f}s)")

    conn.close()
    print(f"\nHoàn tất. Thành công: {n_ok} | Thất bại: {n_fail} | DB: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="engine_sounds_500/manifest_500.csv")
    parser.add_argument("--audio_dir", default="engine_sounds_500/audio")
    parser.add_argument("--db", default="engine_sounds.db")
    args = parser.parse_args()
    main(args.manifest, args.audio_dir, args.db)
