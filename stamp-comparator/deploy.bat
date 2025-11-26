@echo off
setlocal

set ENV=%1
if "%ENV%"=="" set ENV=production

set COMPOSE_FILE=docker-compose.yml
if "%ENV%"=="development" set COMPOSE_FILE=docker-compose.dev.yml

echo ==========================================
echo Stamp Comparator - Deployment Script
echo ==========================================
echo.
echo Deploying in %ENV% mode...
echo.

REM Stop existing containers
echo Stopping existing containers...
docker-compose -f %COMPOSE_FILE% down

REM Pull latest changes (if using git)
if exist ".git" (
    echo Pulling latest changes...
    git pull
)

REM Build containers
echo.
echo Building containers...
docker-compose -f %COMPOSE_FILE% build --no-cache

REM Start containers
echo.
echo Starting containers...
docker-compose -f %COMPOSE_FILE% up -d

REM Wait for services
echo.
echo Waiting for services to start...
timeout /t 15 /nobreak > nul

REM Check backend health
echo Checking backend health...
curl -f http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo Warning: Backend health check failed
    docker-compose -f %COMPOSE_FILE% logs backend
) else (
    echo Backend is healthy
)

REM Check frontend health
echo Checking frontend health...
curl -f http://localhost:3000 > nul 2>&1
if errorlevel 1 (
    echo Warning: Frontend health check failed
    docker-compose -f %COMPOSE_FILE% logs frontend
) else (
    echo Frontend is healthy
)

REM Show running containers
echo.
echo Running containers:
docker-compose -f %COMPOSE_FILE% ps

echo.
echo ==========================================
echo Deployment complete!
echo ==========================================
echo.
echo Application URLs:
echo   Frontend: http://localhost:3000
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo Useful commands:
echo   View logs: docker-compose -f %COMPOSE_FILE% logs -f
echo   Stop services: docker-compose -f %COMPOSE_FILE% down
echo   Restart: docker-compose -f %COMPOSE_FILE% restart
echo.

pause
