@echo off
echo ==========================================
echo Stamp Comparison Tool - Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed!
    pause
    exit /b 1
)
python --version

REM Check Node.js
echo Checking Node.js version...
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed!
    pause
    exit /b 1
)
node --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install backend dependencies
echo.
echo Installing backend dependencies...
if exist "stamp-comparator\requirements.txt" (
    pip install -r stamp-comparator\requirements.txt
) else (
    echo Warning: requirements.txt not found, skipping backend dependencies
)

REM Install frontend dependencies
echo.
echo Installing frontend dependencies...
if exist "stamp-comparator\frontend" (
    cd stamp-comparator\frontend
    call npm install
    cd ..\..
) else (
    echo Warning: frontend directory not found, skipping frontend dependencies
)

REM Create necessary directories
echo.
echo Creating directory structure...
mkdir stamp-comparator\data\reference 2>nul
mkdir stamp-comparator\data\test 2>nul
mkdir stamp-comparator\data\variants 2>nul
mkdir stamp-comparator\data\training 2>nul
mkdir stamp-comparator\models\siamese 2>nul
mkdir stamp-comparator\models\cnn_detector 2>nul
mkdir stamp-comparator\models\autoencoder 2>nul
mkdir stamp-comparator\logs 2>nul

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next Steps:
echo ==========================================
echo.
echo 1. Place your stamp images in stamp-comparator\data\reference\
echo.
echo 2. Prepare training data (if you want to train ML models):
echo    python stamp-comparator\backend\scripts\prepare_training_data.py ^
echo        --source stamp-comparator\data\reference ^
echo        --output stamp-comparator\data\training ^
echo        --variants
echo.
echo 3. Train models (optional - CV methods work without training):
echo    python stamp-comparator\backend\ml_models\train_siamese.py
echo    python stamp-comparator\backend\ml_models\train_cnn_detector.py
echo    python stamp-comparator\backend\ml_models\train_autoencoder.py
echo.
echo 4. Start the backend server:
echo    cd stamp-comparator\backend
echo    uvicorn main:app --reload --port 8000
echo.
echo 5. In a new terminal, start the frontend:
echo    cd stamp-comparator\frontend
echo    npm run dev
echo.
echo 6. Open http://localhost:3000 in your browser
echo.
echo ==========================================
echo.
echo For more information, see stamp-comparator\README.md
echo.

pause
