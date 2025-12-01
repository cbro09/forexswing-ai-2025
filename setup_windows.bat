@echo off
REM ForexSwing AI 2025 - Windows Setup Helper
REM This script helps you prepare files for VM deployment from Windows

echo ==========================================
echo ForexSwing AI 2025 - Windows Setup
echo ==========================================
echo.

:menu
echo.
echo Select an option:
echo 1) Install Python dependencies
echo 2) Create .env file
echo 3) Test bot locally
echo 4) Prepare for VM deployment (create deployment package)
echo 5) Run data collector (one-time)
echo 6) Run API service locally
echo 7) Monitor dashboard
echo 0) Exit
echo.
set /p choice="Enter choice: "

if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto create_env
if "%choice%"=="3" goto test_bot
if "%choice%"=="4" goto prepare_deployment
if "%choice%"=="5" goto collect_data
if "%choice%"=="6" goto run_api
if "%choice%"=="7" goto monitor
if "%choice%"=="0" goto end
goto menu

:install_deps
echo.
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_app.txt
echo.
echo Done! Dependencies installed.
pause
goto menu

:create_env
echo.
if exist .env (
    echo .env file already exists!
    set /p overwrite="Overwrite? (y/N): "
    if /i not "%overwrite%"=="y" goto menu
)
copy .env.example .env
echo.
echo Created .env file from template.
echo.
echo Please edit .env with your API keys:
echo   notepad .env
echo.
echo Required keys:
echo   - ALPHA_VANTAGE_KEY (get at: https://www.alphavantage.co/support/#api-key)
echo   - GOOGLE_API_KEY (get at: https://makersuite.google.com/app/apikey)
echo.
pause
goto menu

:test_bot
echo.
echo Testing ForexBot locally...
python ForexBot.py
echo.
pause
goto menu

:prepare_deployment
echo.
echo Preparing deployment package...
echo.

REM Create deployment directory
if not exist "deployment_package" mkdir deployment_package

REM Copy necessary files
echo Copying files...
xcopy /E /I /Y src deployment_package\src
xcopy /E /I /Y deploy deployment_package\deploy
xcopy /E /I /Y Models deployment_package\Models
if exist data\models xcopy /E /I /Y data\models deployment_package\data\models

copy ForexBot.py deployment_package\
copy companion_api_service.py deployment_package\
copy requirements.txt deployment_package\
copy requirements_app.txt deployment_package\
copy .env.example deployment_package\
copy Dockerfile deployment_package\
copy docker-compose.yml deployment_package\
copy quick_start.sh deployment_package\
copy DEPLOYMENT.md deployment_package\
copy VM_DEPLOYMENT_SUMMARY.md deployment_package\
copy README.md deployment_package\

echo.
echo Deployment package created in: deployment_package\
echo.
echo Next steps:
echo 1. Copy deployment_package to your VM:
echo    scp -r deployment_package user@your-vm-ip:/opt/forexswing-ai-2025
echo.
echo 2. SSH into VM and run:
echo    cd /opt/forexswing-ai-2025
echo    chmod +x deploy/setup_vm.sh quick_start.sh
echo    ./deploy/setup_vm.sh
echo.
pause
goto menu

:collect_data
echo.
echo Running data collector (this may take 15-20 minutes)...
python src\data\market_data_collector.py --once
echo.
echo Data collection complete!
echo Check data\MarketData\ for collected files.
pause
goto menu

:run_api
echo.
echo Starting API service...
echo Access at: http://localhost:8082
echo Press Ctrl+C to stop
echo.
python companion_api_service.py 8082
pause
goto menu

:monitor
echo.
echo Starting monitoring dashboard...
python src\monitoring\dashboard.py
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
