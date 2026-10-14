"""
app_demo.py
===========
Demo giao diện Streamlit, tái hiện đúng sơ đồ khối "Online query phase"
của khung truy hồi audio theo nội dung (CBAR):

    query (audio clip) -> extract features -> similarity search -> ranked results

Chạy:
    streamlit run app_demo.py -- --db engine_sounds.db
"""

import sys
import tempfile
import streamlit as st
import pandas as pd

from database import get_connection, load_all_vectors
from feature_extraction import extract_feature_vector, FEATURE_NAMES
from search import EngineSoundIndex

DB_PATH = "engine_sounds.db"
for i, a in enumerate(sys.argv):
    if a == "--db" and i + 1 < len(sys.argv):
        DB_PATH = sys.argv[i + 1]


@st.cache_resource
def load_index(db_path):
    conn = get_connection(db_path)
    audiooids, vectors, meta = load_all_vectors(conn)
    conn.close()
    return EngineSoundIndex(vectors, meta)


st.set_page_config(page_title="Tìm kiếm tiếng động cơ", layout="wide")
st.title("Hệ thống tìm kiếm tiếng động cơ tương đồng")
st.caption(
    "Kiến trúc: Preprocessing -> Trích đặc trưng (39 chiều) "
    "-> So sánh tương đồng (cosine/L2) -> Top-5 kết quả"
)

index = load_index(DB_PATH)
st.sidebar.write(f"Số file trong CSDL: {len(index.meta)}")
metric = st.sidebar.selectbox("Độ đo tương đồng", ["cosine", "l2", "dot"])
top_k = st.sidebar.slider("Số kết quả trả về", 1, 10, 5)
use_cluster = st.sidebar.checkbox("Dùng tìm kiếm theo cluster (K-means, nhanh hơn)", value=False)

uploaded = st.file_uploader("Tải lên 1 file âm thanh động cơ (WAV)", type=["wav"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner("Đang trích đặc trưng..."):
        q_vec, extra = extract_feature_vector(tmp_path)

    with st.expander("Xem vector đặc trưng đã trích (kết quả trung gian)"):
        df_feat = pd.DataFrame({"đặc trưng": FEATURE_NAMES, "giá trị": q_vec})
        st.dataframe(df_feat, use_container_width=True)

    with st.spinner("Đang tìm kiếm..."):
        if use_cluster:
            results, n_probed = index.query_clustered(q_vec, top_k=top_k, metric=metric)
            st.caption(f"(Chỉ quét {n_probed}/{len(index.meta)} bản ghi nhờ clustering)")
        else:
            results = index.query(q_vec, top_k=top_k, metric=metric)

    st.subheader(f"Top {top_k} kết quả giống nhất")
    for rank, (meta, score) in enumerate(results, start=1):
        cols = st.columns([1, 3, 2, 2, 2])
        cols[0].markdown(f"**#{rank}**")
        cols[1].write(meta["filename"])
        cols[2].write(f"Loại động cơ: {meta.get('engine_label', meta['engine_set'])}")
        cols[3].write(f"Trạng thái: {meta.get('state_label', '-')}")
        cols[4].write(f"Score: {score:.4f}")
        try:
            st.audio(meta["filepath"])
        except Exception:
            pass
        st.divider()
else:
    st.info("Hãy tải lên 1 file WAV để bắt đầu truy vấn.")
