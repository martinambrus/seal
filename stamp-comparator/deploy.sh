#!/bin/bash
set -e

echo "=========================================="
echo "Stamp Comparator - Deployment Script"
echo "=========================================="
echo ""

# Configuration
ENV=${1:-production}
COMPOSE_FILE="docker-compose.yml"

if [ "$ENV" = "development" ]; then
    COMPOSE_FILE="docker-compose.dev.yml"
fi

echo "Deploying in $ENV mode..."
echo ""

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down

# Pull latest changes (if using git)
if [ -d ".git" ]; then
    echo "Pulling latest changes..."
    git pull
fi

# Build containers
echo ""
echo "Building containers..."
docker-compose -f $COMPOSE_FILE build --no-cache

# Start containers
echo ""
echo "Starting containers..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check backend health
echo "Checking backend health..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Backend is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Backend health check failed"
        docker-compose -f $COMPOSE_FILE logs backend
        exit 1
    fi
    sleep 2
done

# Check frontend health
echo "Checking frontend health..."
FRONTEND_PORT=3000
if [ "$ENV" = "production" ]; then
    FRONTEND_PORT=3000
fi

for i in {1..30}; do
    if curl -f http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        echo "✓ Frontend is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Frontend health check failed"
        docker-compose -f $COMPOSE_FILE logs frontend
        exit 1
    fi
    sleep 2
done

# Show running containers
echo ""
echo "Running containers:"
docker-compose -f $COMPOSE_FILE ps

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Application URLs:"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "  Stop services: docker-compose -f $COMPOSE_FILE down"
echo "  Restart: docker-compose -f $COMPOSE_FILE restart"
echo ""
