@echo off
REM Corridor Key Launcher - Local

REM Set script path (assumes corridorkey_cli.py is in the same directory as this batch file)
set "SCRIPT_DIR=%~dp0"
set "LOCAL_SCRIPT=%SCRIPT_DIR%corridorkey_cli.py"

REM SAFETY CHECK: Ensure a folder was dragged onto the script
if "%~1"=="" (
    echo [ERROR] No target folder provided.
    echo.
    echo USAGE: 
    echo Please DRAG AND DROP a folder onto this script to process it.
    echo Do not double-click this script directly.
    echo.
    pause
    exit /b
)

REM Folder dragged? Use it as the target path.
set "WIN_PATH=%~1"

REM Strip trailing slash if present
if "%WIN_PATH:~-1%"=="\" set "WIN_PATH=%WIN_PATH:~0,-1%"

echo Starting Corridor Key locally...
echo Target: "%WIN_PATH%"

REM Run the python script via uv (handles the virtual environment automatically)
uv run python "%LOCAL_SCRIPT%" --action wizard --win_path "%WIN_PATH%" %2 %3 %4 %5 %6 %7 %8 %9

pause
