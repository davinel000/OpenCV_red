@echo off
setlocal

if exist ".venv\\Scripts\\python.exe" (
  echo venv already exists: .venv
  goto :eof
)

python -m venv .venv
if errorlevel 1 (
  echo Failed to create venv. Ensure Python is installed and on PATH.
  exit /b 1
)

call ".venv\\Scripts\\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo venv created and dependencies installed.
