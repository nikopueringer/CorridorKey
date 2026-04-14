@echo off
TITLE CorridorKey Setup Wizard
echo ===================================================
echo     CorridorKey - Windows Auto-Installer
echo ===================================================
echo.

:: 1. Check for uv — install it automatically if missing
where uv >nul 2>&1
if %errorlevel% equ 0 goto :uv_ready

echo [INFO] uv is not installed. Installing now...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install uv. Please visit https://docs.astral.sh/uv/ for manual instructions.
    pause
    exit /b
)

:: uv installer adds to PATH via registry, but the current cmd session
:: doesn't see it yet. Add the default install location so we can continue.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv was installed but cannot be found on PATH.
    echo Please close this window, open a new terminal, and run this script again.
    pause
    exit /b
)
echo [INFO] uv installed successfully.
echo.

:uv_ready

:: 2. Install all dependencies (Python, venv, and packages are handled automatically by uv)
echo [1/2] Installing Dependencies (This might take a while on first run)...
echo       uv will automatically download Python if needed.
uv sync --extra cuda
if %errorlevel% neq 0 (
    echo [ERROR] uv sync failed. Please check the output above for details.
    pause
    exit /b
)

:: 3. Download Weights
echo.
echo [2/2] Downloading CorridorKey Model Weights...
if not exist "CorridorKeyModule\checkpoints" mkdir "CorridorKeyModule\checkpoints"

set "CKPT_DIR=CorridorKeyModule\checkpoints"
set "SAFETENSORS_PATH=%CKPT_DIR%\CorridorKey.safetensors"
set "PTH_PATH=%CKPT_DIR%\CorridorKey.pth"
set "HF_BASE=https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main"

if exist "%SAFETENSORS_PATH%" (
    echo CorridorKey checkpoint already exists!
) else if exist "%PTH_PATH%" (
    echo CorridorKey checkpoint already exists!
) else (
    echo Downloading CorridorKey.safetensors...
    REM --fail returns non-zero on HTTP errors (e.g. 404 before the safetensors
    REM upload lands); fall back to the legacy .pth in that case.
    curl.exe -L --fail -o "%SAFETENSORS_PATH%" "%HF_BASE%/CorridorKey.safetensors"
    if errorlevel 1 (
        echo safetensors not available yet -- falling back to CorridorKey.pth...
        if exist "%SAFETENSORS_PATH%" del "%SAFETENSORS_PATH%"
        REM DEPRECATED: remove after .pth sunset
        curl.exe -L -o "%PTH_PATH%" "%HF_BASE%/CorridorKey_v1.0.pth"
    )
)

echo.
echo ===================================================
echo   Setup Complete! You are ready to key!
echo   Drag and drop folders onto CorridorKey_DRAG_CLIPS_HERE_local.bat
echo ===================================================
pause
