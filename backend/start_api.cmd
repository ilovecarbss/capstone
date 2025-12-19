@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Always run from this script folder
cd /d "%~dp0"

echo ======================================
echo Starting Log Processing API
echo ======================================
echo [Debug] Script dir : %cd%
echo [Debug] .env path  : %cd%\.env
echo.

REM --- Load .env safely (skip blanks/comments) ---
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    set "K=%%A"
    set "V=%%B"
    if not "!K!"=="" (
      if "!K:~0,1!" NEQ "#" (
        REM trim quotes if any
        for %%Q in ("!V!") do set "V=%%~Q"
        set "!K!=!V!"
      )
    )
  )
) else (
  echo [ERROR] .env not found in %cd%
  pause
  exit /b 1
)

echo [Debug] CONDA_BASE=%CONDA_BASE%
echo [Debug] ENV_NAME=%ENV_NAME%
echo [Debug] API_PORT=%API_PORT%
echo.

if "%CONDA_BASE%"=="" (
  echo [ERROR] CONDA_BASE not set in .env
  pause
  exit /b 1
)
set "TRANSFORMERS_NO_TORCHVISION=1"
if "%ENV_NAME%"=="" set "ENV_NAME=loghub"
if "%API_PORT%"=="" set "API_PORT=9000"

set "CONDA_BAT=%CONDA_BASE%\condabin\conda.bat"
if not exist "%CONDA_BAT%" (
  echo [ERROR] conda.bat not found: %CONDA_BAT%
  pause
  exit /b 1
)

echo [Setup] Checking env "%ENV_NAME%" exists...
call "%CONDA_BAT%" env list | findstr /i /c:"%ENV_NAME%" >nul
if errorlevel 1 (
  echo [Setup] Env not found. Creating from environment.yml ...
  call "%CONDA_BAT%" env create -n "%ENV_NAME%" -f environment.yml
  if errorlevel 1 (
    echo [ERROR] Failed to create conda env.
    pause
    exit /b 1
  )
)

echo [Setup] Activating env "%ENV_NAME%"...
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Failed to activate env.
  pause
  exit /b 1
)

echo [Setup] Installing API deps (requirements-api.txt)...
python -m pip install -U -r requirements-api.txt
if errorlevel 1 (
  echo [ERROR] pip install failed
  pause
  exit /b 1
)

echo.
echo [Run] Launching API on http://localhost:%API_PORT%
echo.

REM NOTE: no reload for stability; you can add --reload later
python -u -m uvicorn api:app --host 0.0.0.0 --port %API_PORT% --log-level info

echo.
echo [EXIT] Uvicorn stopped.
pause
