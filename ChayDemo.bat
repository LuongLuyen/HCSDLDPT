@echo off
REM Chay demo he thong tim kiem tieng dong co (Windows)
REM Nhan dup de chay. Dong cua so de dung demo.
cd /d %~dp0
if not exist "venv\Scripts\python.exe" (
  echo [LOI] Chua co moi truong ao venv.
  echo Hay cai dat truoc:
  echo   python -m venv venv
  echo   venv\Scripts\activate
  echo   pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist "engine_sounds.db" (
  echo [LOI] Khong thay engine_sounds.db trong thu muc nay.
  pause
  exit /b 1
)
echo Mo trinh duyet o dia chi hien ra duoi day (thuong la http://localhost:8501)
venv\Scripts\python.exe -m streamlit run app_demo.py
pause
