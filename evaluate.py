"""
evaluate.py
===========
Đánh giá hệ thống tìm kiếm theo đúng định nghĩa Precision/Recall chuẩn
của mô hình lập chỉ mục và truy vấn Boolean:

    Precision = |relevant giao retrieved| / |retrieved|
    Recall    = |relevant giao retrieved| / |relevant|

"Relevant" (liên quan thật) được định nghĩa từ ground-truth có sẵn trong
manifest (KHÔNG cần gán nhãn tay):
    - cùng engine_set (cùng 'loại động cơ'), VÀ
    - |rpm_mean(candidate) - rpm_mean(query)| <= RPM_TOLERANCE
      (cùng 'trạng thái làm việc' ở mức độ gần)

Script so sánh nhiều cấu hình độ đo khác nhau (xếp hạng theo dot product
so với cosine, và khoảng cách Euclid):
    (a) full 39 đặc trưng, metric=cosine
    (b) chỉ MFCC (26 chiều), metric=cosine
    (c) full 39 đặc trưng, metric=l2 (Euclidean)
"""

import numpy as np
from search import EngineSoundIndex
from feature_extraction import FEATURE_NAMES

RPM_TOLERANCE = 400.0  # vòng/phút -- ngưỡng coi là 'cùng trạng thái'


def is_relevant(meta_query, meta_candidate, rpm_tol=RPM_TOLERANCE):
    if meta_candidate["audiooid"] == meta_query["audiooid"]:
        return False  # không tính chính nó là kết quả "tìm thấy"
    same_set = meta_candidate["engine_set"] == meta_query["engine_set"]
    rq, rc = meta_query.get("rpm_mean"), meta_candidate.get("rpm_mean")
    if rq is None or rc is None:
        return same_set  # data moi khong co RPM -> relevant = cung nhom
    close_rpm = abs(rc - rq) <= rpm_tol
    return same_set and close_rpm


def precision_at_k(index, all_vectors, all_meta, k=5, metric="cosine", feature_mask=None):
    """
    Chạy truy vấn "leave-one-out": lần lượt lấy từng file trong CSDL làm query,
    tìm top-k trong CÁC FILE CÒN LẠI, rồi tính precision@k / recall trung bình.
    """
    n = len(all_meta)
    precisions, recalls = [], []

    vectors = all_vectors if feature_mask is None else all_vectors[:, feature_mask]

    for qi in range(n):
        q_raw = vectors[qi]
        q_meta = all_meta[qi]

        # relevant thực sự trong TOÀN BỘ tập (trừ chính nó)
        relevant_ids = {
            all_meta[j]["audiooid"] for j in range(n)
            if is_relevant(q_meta, all_meta[j])
        }
        if len(relevant_ids) == 0:
            continue  # không có bản ghi liên quan nào -> bỏ qua khỏi trung bình

        results = index.query(q_raw, top_k=k, metric=metric)
        retrieved_ids = {m["audiooid"] for m, s in results if m["audiooid"] != q_meta["audiooid"]}

        tp = len(retrieved_ids & relevant_ids)
        precisions.append(tp / max(len(retrieved_ids), 1))
        recalls.append(tp / len(relevant_ids))

    return float(np.mean(precisions)), float(np.mean(recalls))


def run_comparison(all_vectors, all_meta, k=5):
    mfcc_mask = np.array([i for i, name in enumerate(FEATURE_NAMES) if name.startswith("mfcc")])

    configs = [
        ("Full 39 đặc trưng + cosine", None, "cosine"),
        ("Chỉ MFCC (26 chiều) + cosine", mfcc_mask, "cosine"),
        ("Full 39 đặc trưng + L2 (Euclidean)", None, "l2"),
    ]

    print(f"{'Cấu hình':40s} {'Precision@'+str(k):>12s} {'Recall@'+str(k):>10s}")
    print("-" * 64)
    for label, mask, metric in configs:
        vectors = all_vectors if mask is None else all_vectors[:, mask]
        index = EngineSoundIndex(vectors, all_meta)
        p, r = precision_at_k(index, all_vectors, all_meta, k=k, metric=metric, feature_mask=mask)
        print(f"{label:40s} {p:12.3f} {r:10.3f}")


if __name__ == "__main__":
    # smoke test: dữ liệu tổng hợp có cấu trúc (5 'engine_set' x 4 mức rpm)
    np.random.seed(1)
    n_per_group, n_groups, dim = 10, 5, 39
    vectors, meta = [], []
    aoid = 0
    for g in range(n_groups):
        center = np.random.randn(dim) * 3
        for _ in range(n_per_group):
            vectors.append(center + 0.3 * np.random.randn(dim))
            meta.append({
                "audiooid": aoid, "filename": f"g{g}_{aoid}.wav",
                "engine_set": f"set_{g}", "rpm_mean": 2000 + g * 500,
            })
            aoid += 1
    vectors = np.array(vectors)

    run_comparison(vectors, meta, k=5)
