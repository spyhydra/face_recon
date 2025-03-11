@echo off
echo Setting up environment for Windows...

REM Check if VcXsrv is installed
where vcxsrv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo VcXsrv not found. Please install VcXsrv Windows X Server.
    echo Download from: https://sourceforge.net/projects/vcxsrv/
    exit /b 1
)

REM Start VcXsrv if not running
tasklist /FI "IMAGENAME eq vcxsrv.exe" 2>NUL | find /I /N "vcxsrv.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting VcXsrv...
    start "" "C:\Program Files\VcXsrv\vcxsrv.exe" -multiwindow -ac
    timeout /t 3
)

REM Set environment variables for docker-compose
set DISPLAY_ENV=host.docker.internal:0.0

echo Building and starting the application...
docker-compose up --build

echo Application closed. 