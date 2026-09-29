@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  run.bat — Speech Emotion Detection System Launcher
REM  Usage: run.bat [setup|train|backend|frontend|test|all]
REM ═══════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion
set "PROJECT_DIR=%~dp0"
set "PYTHON=python"

echo.
echo  ===================================================================
echo    ^🎙️  Speech Emotion Detection System
echo  ===================================================================
echo.

REM Default action: show menu
set "ACTION=%1"
if "%ACTION%"=="" set "ACTION=menu"

goto :%ACTION% 2>nul || goto :unknown

:menu
echo  Available commands:
echo.
echo    run.bat setup     — Install all dependencies
echo    run.bat dataset   — Check/setup the dataset
echo    run.bat train     — Train the ML model
echo    run.bat backend   — Start the FastAPI backend
echo    run.bat frontend  — Open the frontend in browser
echo    run.bat test      — Run the test suite
echo    run.bat all       — Train + start backend + open browser
echo.
echo  Quick start (after dataset is ready):
echo    1. run.bat setup
echo    2. run.bat train
echo    3. run.bat backend   (in a new terminal)
echo    4. run.bat frontend
echo.
goto :eof

:setup
echo [1/3] Installing Python dependencies...
%PYTHON% -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)
echo.
echo [2/3] Checking dataset...
%PYTHON% "%PROJECT_DIR%src\download_dataset.py"
echo.
echo [3/3] Setup complete!
echo.
echo Next step: run.bat train
pause
goto :eof

:dataset
echo Checking RAVDESS dataset...
%PYTHON% "%PROJECT_DIR%src\download_dataset.py"
pause
goto :eof

:train
echo Starting model training...
echo This may take 10-30 minutes depending on your machine.
echo.
%PYTHON% "%PROJECT_DIR%src\train.py"
if errorlevel 1 (
    echo ERROR: Training failed. Check the error above.
    pause
    exit /b 1
)
echo.
echo Training complete! You can now start the backend.
echo Run: run.bat backend
pause
goto :eof

:backend
echo Starting FastAPI backend on http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.
echo API docs available at: http://127.0.0.1:8000/docs
echo.
%PYTHON% "%PROJECT_DIR%backend\main.py"
goto :eof

:frontend
echo Opening frontend in browser...
start "" "http://127.0.0.1:8000"
echo.
echo If the browser doesn't open automatically, navigate to:
echo http://127.0.0.1:8000
echo.
echo IMPORTANT: Make sure the backend is running first!
echo Run in another terminal: run.bat backend
pause
goto :eof

:test
echo Running test suite...
echo.
%PYTHON% -m pytest "%PROJECT_DIR%tests\" -v --tb=short
pause
goto :eof

:all
echo Starting complete system...
echo.
echo [Step 1] Checking if model exists...
if not exist "%PROJECT_DIR%models\speech_emotion_model.pkl" (
    echo Model not found. Starting training first...
    call :train
    if errorlevel 1 goto :eof
)

echo.
echo [Step 2] Starting backend in background...
start "SED Backend" cmd /c "%PYTHON% %PROJECT_DIR%backend\main.py"

echo Waiting for backend to initialize (5 seconds)...
timeout /t 5 /nobreak > nul

echo.
echo [Step 3] Opening frontend...
start "" "http://127.0.0.1:8000"

echo.
echo ===================================================================
echo  System is running!
echo  Web App:  http://127.0.0.1:8000
echo  API docs: http://127.0.0.1:8000/docs
echo ===================================================================
echo.
echo Close the "SED Backend" window to stop the backend.
pause
goto :eof

:unknown
echo Unknown command: %ACTION%
goto :menu
