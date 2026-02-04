@echo off
setlocal

if not exist ".venv\\Scripts\\python.exe" (
  echo venv not found. Run setup_venv.bat first.
  exit /b 1
)

call ".venv\\Scripts\\activate.bat"
python main.py --config config.yaml %*
