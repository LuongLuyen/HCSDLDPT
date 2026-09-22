"""
select_and_download.py
=======================
Tải 500 file WAV của bộ Procedural Engine Sounds Dataset
(rdoerfler/procedural-engine-sounds) theo đúng danh sách trong
engine_sounds_500/manifest_500.csv.

  - Nếu manifest_500.csv đã có: tải đúng các file được liệt kê trong đó.
  - Nếu chưa có: tự sinh manifest bằng stratified sampling (đều theo 8 set
    "loại động cơ" x 5 mức RPM "trạng thái làm việc"), rồi tải theo danh sách đó.

File nào đã có sẵn sẽ được bỏ qua, nên chạy lại rất nhanh (resume).

Yêu cầu: pip install -r requirements.txt
Chạy trên máy có kết nối tới huggingface.co.
"""

import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "rdoerfler/procedural-engine-sounds"
REPO_TYPE = "dataset"
OUT_DIR = "engine_sounds_500"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest_500.csv")

N_TOTAL = 500
SETS = ["A_full_set", "B_full_set", "C_full_set", "D_full_set",
         "E_large_set", "F_large_set", "G_large_set", "H_large_set"]
BASE_PER_SET = N_TOTAL // len(SETS)        # 62 file/set
EXTRA_SETS = N_TOTAL % len(SETS)           # 4 set đầu lấy thêm 1 file -> đủ 500
N_STATES = 5                               # số mức RPM (trạng thái) muốn phân tầng


def build_manifest():
    """Sinh manifest_500.csv bằng stratified sampling nếu chưa có sẵn."""
    manifest_rows = []
    for si, set_name in enumerate(SETS):
        n_this_set = BASE_PER_SET + (1 if si < EXTRA_SETS else 0)  # 63 hoặc 62

        # 1) Tải file thống kê (rất nhẹ, không phải audio) để biết rpm_mean từng file
        stats_path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=f"dataset/metadata/{set_name}_stats.csv",
        )
        df = pd.read_csv(stats_path)

        # 2) Chia file trong set thành N_STATES nhóm theo rpm_mean (idle -> full rpm)
        df = df.sort_values("rpm_mean").reset_index(drop=True)
        df["state_bin"] = pd.qcut(df["rpm_mean"], N_STATES, labels=False, duplicates="drop")

        # 3) Lấy đều số file từ mỗi bin để tổng = n_this_set
        per_bin = max(1, n_this_set // N_STATES)
        # (không dùng groupby.apply vì pandas >= 2.2 loại bỏ cột group key khỏi kết quả)
        picked = pd.concat([
            g.sample(n=min(per_bin, len(g)), random_state=42)
            for _, g in df.groupby("state_bin", group_keys=False)
        ])
        # nếu còn thiếu do làm tròn, bù thêm ngẫu nhiên từ phần chưa chọn
        remaining = n_this_set - len(picked)
        if remaining > 0:
            leftover = df.drop(picked.index)
            picked = pd.concat([picked, leftover.sample(n=min(remaining, len(leftover)), random_state=42)])

        picked["engine_set"] = set_name
        manifest_rows.append(picked)

    manifest = pd.concat(manifest_rows, ignore_index=True)
    manifest["state_label"] = manifest["state_bin"].map({
        0: "idle_lowest_rpm", 1: "low", 2: "mid", 3: "high", 4: "very_high_rpm"
    })
    return manifest


def download_audio(manifest):
    """Tải đúng các file WAV được liệt kê trong manifest (bỏ qua file đã có)."""
    n_new = 0
    for _, row in manifest.iterrows():
        set_name = row["engine_set"]
        fname = row["filename"]
        dest = os.path.join(AUDIO_DIR, f"{set_name}__{fname}")
        if os.path.exists(dest):
            continue  # đã có sẵn -> khỏi gọi mạng (tránh HEAD request cho từng file)
        local_path = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=f"dataset/audio/{set_name}/{fname}",
        )
        os.symlink(local_path, dest) if hasattr(os, "symlink") else \
            __import__("shutil").copy(local_path, dest)
        n_new += 1
    return n_new


if __name__ == "__main__":
    os.makedirs(AUDIO_DIR, exist_ok=True)

    if os.path.exists(MANIFEST_PATH):
        manifest = pd.read_csv(MANIFEST_PATH)
        print(f"Dùng danh sách có sẵn: {MANIFEST_PATH} ({len(manifest)} file)")
    else:
        print("Chưa có manifest, sinh danh sách bằng stratified sampling...")
        manifest = build_manifest()
        manifest.to_csv(MANIFEST_PATH, index=False)

    n_new = download_audio(manifest)
    print(f"Đã tải {n_new} file mới (bỏ qua {len(manifest) - n_new} file đã có).")
    print(f"Tổng: {len(manifest)} file | danh sách: {MANIFEST_PATH}")
    print(manifest["engine_set"].value_counts())
    print(manifest["state_label"].value_counts())
