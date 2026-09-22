#!/usr/bin/env bash
# setup_env.sh
# =============
# Tạo môi trường ảo (virtual environment) cho dự án, cài đặt toàn bộ
# thư viện cần thiết (huggingface_hub, librosa, scikit-learn, streamlit...).
#
# Cách dùng:
#   bash setup_env.sh
#   source venv/bin/activate      # (Linux/Mac)
#   venv\Scripts\activate         # (Windows, chạy trong cmd/powershell)
#
# Sau khi kích hoạt môi trường ảo, chạy lần lượt các bước trong README.md.

set -e

# Chọn Python tương thích: llvmlite/numba (librosa cần) không phát hành wheel
# cho Python quá mới (vd 3.14) trên macOS x86_64 -> ưu tiên 3.10-3.12.
if [ -z "${PYTHON_BIN:-}" ]; then
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            v="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
            case "$v" in
                3.10|3.11|3.12) PYTHON_BIN="$cand"; break;;
            esac
        fi
    done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="venv"

echo ">> Kiểm tra Python..."
$PYTHON_BIN --version

echo ">> Tạo virtual environment tại ./$VENV_DIR ..."
$PYTHON_BIN -m venv "$VENV_DIR"

# Kích hoạt venv ngay trong script này để cài đặt thư viện
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"        # Linux / Mac
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"    # Git-Bash trên Windows
else
    echo "Không tìm thấy file activate, kiểm tra lại venv."
    exit 1
fi

echo ">> Nâng cấp pip..."
pip install --upgrade pip

echo ">> Cài đặt thư viện từ requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "=========================================================="
echo "  Môi trường ảo đã sẵn sàng."
echo "  Kích hoạt lại (ở các lần sau) bằng:"
echo "    source $VENV_DIR/bin/activate       (Linux/Mac)"
echo "    $VENV_DIR\\Scripts\\activate          (Windows)"
echo "=========================================================="
