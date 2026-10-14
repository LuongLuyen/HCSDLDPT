"""
engine_labels.py
================
2 bảng tên hiển thị:
(1) ENGINE_SET_LABELS: 8 set gốc procedural A-H (3 máy E1/E2/E3) - DB cũ.
(2) NHOM8_LABELS: 8 nhóm mới theo đề tài máy nổ Việt Nam (A-H).
    Nhãn proxy (= dùng tạm thay thế) ghi ngay trong tên hiển thị,
    chi tiết xem dataset/DanhSach.csv và Báo cáo (Bảng 2.1).
"""

ENGINE_SET_LABELS = {
    # Máy 4 xi-lanh (E1) - xe con phổ thông
    "A_full_set": "Máy xăng 4 xi-lanh turbo - thu trong xe (ghế lái)",
    # Máy V8 gầm cao (E2) - 2 vị trí thu
    "B_full_set": "Máy V8 gầm cao - thu dưới sàn xe",
    "H_large_set": "Máy V8 gầm cao - thu ở táp-lô",
    # Máy V8 thể thao (E3) - 5 vị trí thu
    "C_full_set": "Máy V8 thể thao - thu trong xe (ghế lái)",
    "D_full_set": "Máy V8 thể thao - thu mic trầm trước đầu xe",
    "E_large_set": "Máy V8 thể thao - thu mic trầm (bản large)",
    "F_large_set": "Máy V8 thể thao - thu mono trong cabin",
    "G_large_set": "Máy V8 thể thao - thu ngoài ống xả",
}

ENGINE_SET_SHORT = {
    "A_full_set": "Máy 4 xi-lanh",
    "B_full_set": "Máy V8 gầm cao",
    "C_full_set": "Máy V8 thể thao",
    "D_full_set": "Máy V8 thể thao",
    "E_large_set": "Máy V8 thể thao",
    "F_large_set": "Máy V8 thể thao",
    "G_large_set": "Máy V8 thể thao (ống xả)",
    "H_large_set": "Máy V8 gầm cao",
}


def get_label(engine_set):
    return ENGINE_SET_LABELS.get(engine_set, NHOM8_LABELS.get(engine_set, engine_set))


# Bảng 8 nhóm mới (dataset/DanhSach.csv) - tên gần gũi tiếng Việt
NHOM8_LABELS = {
    "A": "A - Động cơ xe máy (máy 2 kỳ đại diện, proxy)",
    "B": "B - Động cơ ô tô xăng (Zenodo, thật)",
    "C": "C - Động cơ diesel / máy dầu (Zenodo, thật)",
    "D": "D - Máy phát điện (diesel ổn định đại diện, proxy)",
    "E": "E - Máy nông nghiệp (cưa xích/đầu nổ đại diện, proxy)",
    "F": "F - Động cơ tàu thuyền (máy xăng đại diện, proxy)",
    "G": "G - Máy công trình (V8 chịu tải đại diện, proxy)",
    "H": "H - Máy công nghiệp (tổng hợp chuyên dụng, proxy)",
}
