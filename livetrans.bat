@echo off
rem ---------------------------------------------------------------
rem  IMPORTANT: keep this file pure ASCII.
rem  cmd parses .bat files with the system ANSI code page (GBK on a
rem  Chinese Windows), while source files are UTF-8. Chinese text makes
rem  the byte stream misalign, the batch line structure breaks, and cmd
rem  starts executing fragments as commands. So the interactive menu
rem  lives in livetrans/menu.py, where UTF-8 is handled properly.
rem ---------------------------------------------------------------

cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo   [x] Virtual environment not found: %PY%
    echo.
    echo       Create it in this folder first:
    echo           uv venv --python 3.12 .venv
    echo           uv pip install --python .venv faster-whisper soundcard numpy PySide6 httpx
    echo.
    pause
    exit /b 1
)

rem With arguments they are handed straight to livetrans:
rem     livetrans.bat --terminal
rem     livetrans.bat --devices 8
rem     livetrans.bat --check
"%PY%" -m livetrans.menu %*

exit /b %errorlevel%
