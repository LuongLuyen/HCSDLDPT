"""
search.py
=========
Hệ tìm kiếm tương đồng, bám sát công thức Lecture 7 - Vector space
database and Clustering:

  - Dot product        : S(Di,Qj) = sum_k Di_k * Qj_k                (slide 6)
  - Cosine similarity   : S = (Di . Qj) / (|Di| * |Qj|)               (slide 8)
  - L2 / Euclidean dist : L2(Di,Dj) = sqrt(sum_k (dik - djk)^2)       (slide 22)
  - Clustered search    : K-means trước, chỉ probe các cluster gần nhất
    để tăng tốc (slide "Querying Clustered Vector Data")

Đặc trưng ở đây có thang đo rất khác nhau (E ~ 0.1, centroid ~ 2000Hz,
MFCC ~ -100..50) nên BẮT BUỘC chuẩn hóa (z-score) trước khi tính
similarity -- nếu không đặc trưng có biên độ lớn (centroid, MFCC) sẽ
lấn át các đặc trưng khác. Đây là mở rộng thực tế cho nhược điểm đã
nêu trong Lecture 7 slide 12: "vector-space query model không tính đến
sự khác biệt về ý nghĩa giữa các thuộc tính".
"""

import numpy as np
from sklearn.cluster import KMeans


class EngineSoundIndex:
    def __init__(self, vectors, meta):
        self.meta = meta
        self.mean = vectors.mean(axis=0)
        self.std = vectors.std(axis=0) + 1e-8
        self.vectors_norm = (vectors - self.mean) / self.std   # z-score chuẩn hóa
        self.kmeans = None

    # ---------- độ đo tương đồng / khoảng cách (Lecture 7) ----------
    @staticmethod
    def dot_product(d, q):
        return np.dot(d, q)

    @staticmethod
    def cosine_similarity(d, q):
        num = np.dot(d, q)
        den = np.linalg.norm(d) * np.linalg.norm(q) + 1e-10
        return num / den

    @staticmethod
    def l2_distance(d, q):
        return np.sqrt(np.sum((d - q) ** 2))

    def _normalize_query(self, q_raw):
        return (q_raw - self.mean) / self.std

    # ---------- truy vấn tuần tự (không cluster) ----------
    def query(self, q_raw, top_k=5, metric="cosine"):
        """
        metric: 'cosine' | 'dot' | 'l2'
        Trả về list [(meta_i, score_i), ...] sắp xếp giảm dần độ tương đồng
        (với 'l2' thì sắp xếp tăng dần khoảng cách, trả về dạng score = -distance
        để thống nhất "cao hơn = tốt hơn").
        """
        q = self._normalize_query(q_raw)
        scores = []
        for i, d in enumerate(self.vectors_norm):
            if metric == "cosine":
                s = self.cosine_similarity(d, q)
            elif metric == "dot":
                s = self.dot_product(d, q)
            elif metric == "l2":
                s = -self.l2_distance(d, q)
            else:
                raise ValueError("metric phải là 'cosine', 'dot' hoặc 'l2'")
            scores.append(s)

        order = np.argsort(scores)[::-1][:top_k]
        return [(self.meta[i], scores[i]) for i in order]

    # ---------- clustering (Lecture 7: K-means / Querying Clustered Vector Data) ----------
    def build_clusters(self, n_clusters=8, random_state=42):
        """
        n_clusters=8 mặc định vì dataset có đúng 8 'loại động cơ' (set A-H)
        -- kỳ vọng K-means sẽ tự nhiên tách theo loại động cơ (Lecture 7:
        'Cluster centroids có thể dùng làm index entry points').
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.vectors_norm)
        return self.cluster_labels

    def query_clustered(self, q_raw, top_k=5, n_probe=2, metric="cosine"):
        """
        Bước 1: tính khoảng cách từ Q đến n_probe centroid gần nhất (multi-probe,
                slide "Querying Clustered Vector Data" -- tránh bỏ sót biên giới).
        Bước 2: chỉ tính similarity chi tiết với các điểm trong các cluster đó.
        """
        if self.kmeans is None:
            self.build_clusters()

        q = self._normalize_query(q_raw)
        centroid_dists = np.linalg.norm(self.kmeans.cluster_centers_ - q, axis=1)
        nearest_clusters = np.argsort(centroid_dists)[:n_probe]

        candidate_idx = [i for i, c in enumerate(self.cluster_labels) if c in nearest_clusters]

        scores = []
        for i in candidate_idx:
            d = self.vectors_norm[i]
            if metric == "cosine":
                s = self.cosine_similarity(d, q)
            elif metric == "l2":
                s = -self.l2_distance(d, q)
            else:
                s = self.dot_product(d, q)
            scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]
        return [(self.meta[i], s) for i, s in top], len(candidate_idx)


if __name__ == "__main__":
    # smoke test với dữ liệu ngẫu nhiên
    np.random.seed(0)
    n, dim = 100, 39
    vecs = np.random.randn(n, dim)
    meta = [{"filename": f"file_{i}.wav", "engine_set": f"set_{i%8}"} for i in range(n)]

    idx = EngineSoundIndex(vecs, meta)
    q = vecs[10] + 0.01 * np.random.randn(dim)  # truy vấn gần giống file 10

    print("== Cosine (tuần tự) ==")
    for m, s in idx.query(q, top_k=5, metric="cosine"):
        print(f"  {m['filename']:15s} score={s:.4f}")

    print("== Clustered (K-means, probe 2 cluster) ==")
    top, n_cand = idx.query_clustered(q, top_k=5, n_probe=2)
    print(f"  (chỉ xét {n_cand}/{n} bản ghi thay vì quét toàn bộ)")
    for m, s in top:
        print(f"  {m['filename']:15s} score={s:.4f}")
